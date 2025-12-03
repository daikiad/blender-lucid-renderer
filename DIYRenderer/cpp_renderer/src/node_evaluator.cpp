#include "renderer.hpp"
#include <iostream>

/**
 * ========== Node Graph Evaluation ==========
 * 
 * This module acts as a "shader compiler" for Blender's node-based materials.
 * It recursively evaluates node graphs to extract material properties like
 * albedo (Base Color) and emission.
 * 
 * Key Design:
 * - Nodes are evaluated on-demand during ray tracing
 * - Socket connections are followed recursively (depth-first traversal)
 * - Default values are used when sockets are not connected
 * - Supports: Principled BSDF, Emission, RGB, Mix RGB nodes
 * 
 * CRITICAL BUG FIX:
 * SocketValue union must use correct field for each type:
 * - VEC4 → use v4 field (RGBA colors)
 * - VEC3 → use v3 field (RGB colors, vectors)
 * - FLOAT → use f field (scalars)
 * 
 * Previous bug: Code always read v3 field even for VEC4 types,
 * causing all materials to render black. Fixed by checking type enum.
 */

/**
 * evaluateNode: Recursively evaluate a node's output socket
 * 
 * @param tree The node tree containing all nodes
 * @param nodeName Name of the node to evaluate
 * @param socketName Name of the output socket to read
 * @return RGB color value from the socket
 * 
 * This function follows socket connections recursively:
 * 1. Find the requested node by name
 * 2. Check if the input socket is connected
 * 3. If connected: recursively evaluate the linked node
 * 4. If not connected: use the socket's default value
 */
Vec3 evaluateNode(const NodeTree &tree, const std::string &nodeName, const std::string &socketName) {
    const MaterialNode *node = tree.findNode(nodeName);
    if(!node) {
        std::cerr << "[NodeEval] Node not found: " << nodeName << "\n";
        return Vec3(1.0f, 0.0f, 1.0f);  // Magenta = error
    }
    
    // ===== Principled BSDF Node =====
    if(node->type == "ShaderNodeBsdfPrincipled") {
        // Principled BSDF node
        if(socketName == "BSDF") {
            // For BSDF output, we need to evaluate the Base Color input
            const NodeSocket *baseColorSocket = node->findInput("Base Color");
            
            static int debugCount = 0;
            if(++debugCount <= 2) {
                std::cerr << "[NodeEval] Principled BSDF / BSDF: baseColorSocket=" << (baseColorSocket ? "found" : "NULL") << "\n";
                if(baseColorSocket) {
                    std::cerr << "[NodeEval]   is_linked=" << baseColorSocket->is_linked 
                              << " type=" << (int)baseColorSocket->default_value.type << "\n";
                    // Always print v3 values regardless of type check
                    std::cerr << "[NodeEval]   v3 union contents=(" << baseColorSocket->default_value.v3.x 
                              << "," << baseColorSocket->default_value.v3.y 
                              << "," << baseColorSocket->default_value.v3.z << ")\n";
                    std::cerr << "[NodeEval]   VEC3 enum=" << (int)SocketValue::VEC3 
                              << " VEC4 enum=" << (int)SocketValue::VEC4 << "\n";
                }
            }
            
            if(!baseColorSocket) {
                return Vec3(0.8f, 0.8f, 0.8f);  // Default gray
            }
            
            if(baseColorSocket->is_linked) {
                // Follow connection
                return evaluateNode(tree, baseColorSocket->linked_node, baseColorSocket->linked_socket);
            } else {
                // Use default value
                if(baseColorSocket->default_value.type == SocketValue::VEC4) {
                    return baseColorSocket->default_value.v4;  // VEC4 uses v4 field
                } else if(baseColorSocket->default_value.type == SocketValue::VEC3) {
                    return baseColorSocket->default_value.v3;  // VEC3 uses v3 field
                }
                return Vec3(0.8f, 0.8f, 0.8f);
            }
        } else if(socketName == "Base Color") {
            // Direct query of Base Color
            const NodeSocket *socket = node->findInput("Base Color");
            if(socket && !socket->is_linked) {
                if(socket->default_value.type == SocketValue::VEC4) {
                    return socket->default_value.v4;
                } else if(socket->default_value.type == SocketValue::VEC3) {
                    return socket->default_value.v3;
                }
            }
            return Vec3(0.8f, 0.8f, 0.8f);
        }
    } else if(node->type == "ShaderNodeEmission") {
        // ===== Emission Node =====
        // Emission shader: emits light (Color * Strength)
        if(socketName == "Emission") {
            const NodeSocket *colorSocket = node->findInput("Color");
            const NodeSocket *strengthSocket = node->findInput("Strength");
            
            Vec3 color(1.0f, 1.0f, 1.0f);  // Default white
            float strength = 1.0f;          // Default strength
            
            // Evaluate color (can be connected or constant)
            if(colorSocket) {
                if(colorSocket->is_linked) {
                    color = evaluateNode(tree, colorSocket->linked_node, colorSocket->linked_socket);
                } else if(colorSocket->default_value.type == SocketValue::VEC4) {
                    color = colorSocket->default_value.v4;  // RGBA → use v4 field
                } else if(colorSocket->default_value.type == SocketValue::VEC3) {
                    color = colorSocket->default_value.v3;  // RGB → use v3 field
                }
            }
            
            // Get strength (usually a constant float)
            if(strengthSocket && !strengthSocket->is_linked) {
                if(strengthSocket->default_value.type == SocketValue::FLOAT) {
                    strength = strengthSocket->default_value.f;
                }
            }
            
            return color * strength;  // Multiply color by strength
        }
    } else if(node->type == "ShaderNodeRGB") {
        // ===== RGB Node =====
        // Simple constant color node (like a color picker in Blender)
        for(const auto &output : node->outputs) {
            if(output.name == "Color" && output.default_value.type != SocketValue::NONE) {
                if(output.default_value.type == SocketValue::VEC4) {
                    return output.default_value.v4;  // RGBA → use v4 field
                } else if(output.default_value.type == SocketValue::VEC3) {
                    return output.default_value.v3;  // RGB → use v3 field
                }
            }
        }
        // Fallback: return white if no valid color found
        return Vec3(1.0f, 1.0f, 1.0f);
    } else if(node->type == "ShaderNodeMix" || node->type == "ShaderNodeMixRGB") {
        // ===== Mix Node =====
        // Blends two colors using various blend modes (Mix, Add, Multiply, etc.)
        const NodeSocket *facSocket = node->findInput("Fac");
        const NodeSocket *aSocket = node->findInput("A");
        const NodeSocket *bSocket = node->findInput("B");
        
        float fac = 0.5f;
        Vec3 colorA(0, 0, 0), colorB(1, 1, 1);
        
        if(facSocket && !facSocket->is_linked) {
            if(facSocket->default_value.type == SocketValue::FLOAT) {
                fac = facSocket->default_value.f;
            }
        }
        
        if(aSocket) {
            if(aSocket->is_linked) {
                colorA = evaluateNode(tree, aSocket->linked_node, aSocket->linked_socket);
            } else if(aSocket->default_value.type == SocketValue::VEC4) {
                colorA = aSocket->default_value.v4;
            } else if(aSocket->default_value.type == SocketValue::VEC3) {
                colorA = aSocket->default_value.v3;
            }
        }
        
        if(bSocket) {
            if(bSocket->is_linked) {
                colorB = evaluateNode(tree, bSocket->linked_node, bSocket->linked_socket);
            } else if(bSocket->default_value.type == SocketValue::VEC4) {
                colorB = bSocket->default_value.v4;
            } else if(bSocket->default_value.type == SocketValue::VEC3) {
                colorB = bSocket->default_value.v3;
            }
        }
        
        // Linear interpolation: lerp(A, B, fac) = A * (1 - fac) + B * fac
        return colorA * (1.0f - fac) + colorB * fac;
    } else if(node->type == "ShaderNodeOutputMaterial") {
        // ===== Material Output Node =====
        // This is the final output node - traverse to Surface input
        const NodeSocket *surfaceSocket = node->findInput("Surface");
        if(surfaceSocket && surfaceSocket->is_linked) {
            return evaluateNode(tree, surfaceSocket->linked_node, surfaceSocket->linked_socket);
        }
    }
    
    // Unknown node type - return magenta error color
    std::cerr << "[NodeEval] Unhandled node type: " << node->type << " socket: " << socketName << "\n";
    return Vec3(1.0f, 0.0f, 1.0f);  // Magenta = unimplemented
}

/**
 * getAlbedoFromNodeTree: Extract base color (diffuse albedo) from node graph
 * 
 * @param tree The material node tree to evaluate
 * @return RGB albedo color (diffuse reflectance)
 * 
 * Algorithm:
 * 1. Find Material Output node (the final node)
 * 2. Follow its Surface input connection
 * 3. If connected to Principled BSDF, get its Base Color
 * 4. Recursively evaluate any connections on Base Color
 * 
 * This is called once per ray-surface intersection to determine
 * the surface's diffuse color for path tracing calculations.
 */
Vec3 getAlbedoFromNodeTree(const NodeTree &tree) {
    static int callCount = 0;
    if(++callCount <= 3) {
        std::cerr << "[NodeEval] getAlbedoFromNodeTree called, valid=" << tree.valid << "\n";
    }
    
    if(!tree.valid) {
        std::cerr << "[NodeEval] Tree not valid, returning gray\n";
        return Vec3(0.8f, 0.8f, 0.8f);  // Default gray
    }
    
    // Find Material Output node (entry point)
    const MaterialNode *outputNode = tree.findOutputNode();
    if(!outputNode) {
        std::cerr << "[NodeEval] No Material Output node found\n";
        return Vec3(0.8f, 0.8f, 0.8f);
    }
    
    // Get Surface input (should be connected to a shader like Principled BSDF)
    const NodeSocket *surfaceSocket = outputNode->findInput("Surface");
    if(!surfaceSocket || !surfaceSocket->is_linked) {
        std::cerr << "[NodeEval] Surface socket not connected\n";
        return Vec3(0.8f, 0.8f, 0.8f);
    }
    
    if(callCount <= 3) {
        std::cerr << "[NodeEval] Evaluating: " << surfaceSocket->linked_node << " / " << surfaceSocket->linked_socket << "\n";
    }
    
    // Evaluate connected BSDF node
    Vec3 result = evaluateNode(tree, surfaceSocket->linked_node, surfaceSocket->linked_socket);
    
    if(callCount <= 3) {
        std::cerr << "[NodeEval] Result: (" << result.x << "," << result.y << "," << result.z << ")\n";
    }
    
    return result;
}

/**
 * getEmissionFromNodeTree: Extract emission (light) from node graph
 * 
 * @param tree The material node tree to evaluate
 * @return RGB emission color (light emitted by surface)
 * 
 * Algorithm:
 * 1. Find Material Output node
 * 2. Follow its Surface input connection
 * 3. Check if connected to Emission shader
 * 4. If Principled BSDF, check for Emission Color socket
 * 5. Return (0,0,0) if no emission found
 * 
 * This determines if a surface emits light (acts as a light source).
 * Used to implement area lights in path tracing.
 */
Vec3 getEmissionFromNodeTree(const NodeTree &tree) {
    if(!tree.valid) {
        return Vec3(0.0f, 0.0f, 0.0f);  // No emission
    }
    
    // Find Material Output node
    const MaterialNode *outputNode = tree.findOutputNode();
    if(!outputNode) {
        return Vec3(0.0f, 0.0f, 0.0f);
    }
    
    // Get Surface input
    const NodeSocket *surfaceSocket = outputNode->findInput("Surface");
    if(!surfaceSocket || !surfaceSocket->is_linked) {
        return Vec3(0.0f, 0.0f, 0.0f);
    }
    
    // Check if connected node is Emission shader
    const MaterialNode *shaderNode = tree.findNode(surfaceSocket->linked_node);
    if(shaderNode && shaderNode->type == "ShaderNodeEmission") {
        return evaluateNode(tree, surfaceSocket->linked_node, "Emission");
    }
    
    // Principled BSDF also supports emission (for glowing objects)
    if(shaderNode && shaderNode->type == "ShaderNodeBsdfPrincipled") {
        // Try both "Emission Color" (newer Blender) and "Emission" (older versions)
        const NodeSocket *emissionSocket = shaderNode->findInput("Emission Color");
        if(!emissionSocket) {
            emissionSocket = shaderNode->findInput("Emission");
        }
        
        if(emissionSocket) {
            if(emissionSocket->is_linked) {
                Vec3 emissionColor = evaluateNode(tree, emissionSocket->linked_node, emissionSocket->linked_socket);
                // Apply emission strength
                const NodeSocket *strengthSocket = shaderNode->findInput("Emission Strength");
                float strength = 0.0f;
                if(strengthSocket && strengthSocket->default_value.type == SocketValue::FLOAT) {
                    strength = strengthSocket->default_value.f;
                }
                return emissionColor * strength;
            } else if(emissionSocket->default_value.type == SocketValue::VEC4) {
                // Emission strength is separate in Principled BSDF
                const NodeSocket *strengthSocket = shaderNode->findInput("Emission Strength");
                float strength = 0.0f;
                if(strengthSocket && strengthSocket->default_value.type == SocketValue::FLOAT) {
                    strength = strengthSocket->default_value.f;
                }
                return emissionSocket->default_value.v4 * strength;
            } else if(emissionSocket->default_value.type == SocketValue::VEC3) {
                // Emission strength is separate in Principled BSDF
                const NodeSocket *strengthSocket = shaderNode->findInput("Emission Strength");
                float strength = 0.0f;
                if(strengthSocket && strengthSocket->default_value.type == SocketValue::FLOAT) {
                    strength = strengthSocket->default_value.f;
                }
                return emissionSocket->default_value.v3 * strength;
            }
        }
    }
    
    return Vec3(0.0f, 0.0f, 0.0f);
}

/**
 * getTransmissionFromNodeTree: Extract transmission (glass/transparency) from node graph
 * 
 * @param tree The material node tree to evaluate
 * @return Transmission value (0.0 = opaque, 1.0 = fully transparent)
 * 
 * Algorithm:
 * 1. Find Material Output node
 * 2. Follow its Surface input connection
 * 3. If connected to Principled BSDF, get its Transmission socket
 */
float getTransmissionFromNodeTree(const NodeTree &tree) {
    if(!tree.valid) {
        return 0.0f;  // No transmission (opaque)
    }
    
    // Find Material Output node
    const MaterialNode *outputNode = tree.findOutputNode();
    if(!outputNode) {
        return 0.0f;
    }
    
    // Get Surface input
    const NodeSocket *surfaceSocket = outputNode->findInput("Surface");
    if(!surfaceSocket || !surfaceSocket->is_linked) {
        return 0.0f;
    }
    
    // Check if connected to Principled BSDF
    const MaterialNode *shaderNode = tree.findNode(surfaceSocket->linked_node);
    if(shaderNode && shaderNode->type == "ShaderNodeBsdfPrincipled") {
        // Try "Transmission" socket (Blender 4.0+)
        const NodeSocket *transmissionSocket = shaderNode->findInput("Transmission");
        // Older Blender versions might use "Transmission Weight"
        if(!transmissionSocket) {
            transmissionSocket = shaderNode->findInput("Transmission Weight");
        }
        
        if(transmissionSocket && !transmissionSocket->is_linked) {
            if(transmissionSocket->default_value.type == SocketValue::FLOAT) {
                return transmissionSocket->default_value.f;
            }
        }
    }
    
    return 0.0f;
}

/**
 * getIORFromNodeTree: Extract Index of Refraction from node graph
 * 
 * @param tree The material node tree to evaluate
 * @return IOR value (1.0 = air, 1.45 = glass, 1.33 = water, 2.42 = diamond)
 * 
 * Algorithm:
 * 1. Find Material Output node
 * 2. Follow its Surface input connection
 * 3. If connected to Principled BSDF, get its IOR socket
 */
float getIORFromNodeTree(const NodeTree &tree) {
    if(!tree.valid) {
        return 1.45f;  // Default glass IOR
    }
    
    // Find Material Output node
    const MaterialNode *outputNode = tree.findOutputNode();
    if(!outputNode) {
        return 1.45f;
    }
    
    // Get Surface input
    const NodeSocket *surfaceSocket = outputNode->findInput("Surface");
    if(!surfaceSocket || !surfaceSocket->is_linked) {
        return 1.45f;
    }
    
    // Check if connected to Principled BSDF
    const MaterialNode *shaderNode = tree.findNode(surfaceSocket->linked_node);
    if(shaderNode && shaderNode->type == "ShaderNodeBsdfPrincipled") {
        const NodeSocket *iorSocket = shaderNode->findInput("IOR");
        
        if(iorSocket && !iorSocket->is_linked) {
            if(iorSocket->default_value.type == SocketValue::FLOAT) {
                return iorSocket->default_value.f;
            }
        }
    }
    
    return 1.45f;  // Default glass IOR
}
