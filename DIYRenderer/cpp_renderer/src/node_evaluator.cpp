#include "renderer.hpp"
#include <iostream>

// ========== Node Graph Evaluation ==========
// This acts as a "shader compiler" for Blender's node system
// It evaluates node graphs to get material properties

// Evaluate a single node output socket (recursive for connected inputs)
Vec3 evaluateNode(const NodeTree &tree, const std::string &nodeName, const std::string &socketName) {
    const MaterialNode *node = tree.findNode(nodeName);
    if(!node) {
        std::cerr << "[NodeEval] Node not found: " << nodeName << "\n";
        return Vec3(1.0f, 0.0f, 1.0f);  // Magenta = error
    }
    
    // Handle different node types
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
        // Emission node
        if(socketName == "Emission") {
            const NodeSocket *colorSocket = node->findInput("Color");
            const NodeSocket *strengthSocket = node->findInput("Strength");
            
            Vec3 color(1.0f, 1.0f, 1.0f);
            float strength = 1.0f;
            
            if(colorSocket) {
                if(colorSocket->is_linked) {
                    color = evaluateNode(tree, colorSocket->linked_node, colorSocket->linked_socket);
                } else if(colorSocket->default_value.type == SocketValue::VEC4) {
                    color = colorSocket->default_value.v4;
                } else if(colorSocket->default_value.type == SocketValue::VEC3) {
                    color = colorSocket->default_value.v3;
                }
            }
            
            if(strengthSocket && !strengthSocket->is_linked) {
                if(strengthSocket->default_value.type == SocketValue::FLOAT) {
                    strength = strengthSocket->default_value.f;
                }
            }
            
            return color * strength;
        }
    } else if(node->type == "ShaderNodeRGB") {
        // RGB node (constant color)
        for(const auto &output : node->outputs) {
            if(output.name == "Color" && output.default_value.type != SocketValue::NONE) {
                if(output.default_value.type == SocketValue::VEC4) {
                    return output.default_value.v4;
                } else if(output.default_value.type == SocketValue::VEC3) {
                    return output.default_value.v3;
                }
            }
        }
        // Fallback: check properties for 'color' value
        // (In real implementation, we'd need to store node properties separately)
        return Vec3(1.0f, 1.0f, 1.0f);
    } else if(node->type == "ShaderNodeMix" || node->type == "ShaderNodeMixRGB") {
        // Mix node (blend two inputs)
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
        
        // Linear interpolation
        return colorA * (1.0f - fac) + colorB * fac;
    } else if(node->type == "ShaderNodeOutputMaterial") {
        // Material Output - traverse to Surface input
        const NodeSocket *surfaceSocket = node->findInput("Surface");
        if(surfaceSocket && surfaceSocket->is_linked) {
            return evaluateNode(tree, surfaceSocket->linked_node, surfaceSocket->linked_socket);
        }
    }
    
    std::cerr << "[NodeEval] Unhandled node type: " << node->type << " socket: " << socketName << "\n";
    return Vec3(1.0f, 0.0f, 1.0f);  // Magenta = unimplemented
}

// Get albedo (base color) from node tree
Vec3 getAlbedoFromNodeTree(const NodeTree &tree) {
    static int callCount = 0;
    if(++callCount <= 3) {
        std::cerr << "[NodeEval] getAlbedoFromNodeTree called, valid=" << tree.valid << "\n";
    }
    
    if(!tree.valid) {
        std::cerr << "[NodeEval] Tree not valid, returning gray\n";
        return Vec3(0.8f, 0.8f, 0.8f);  // Default gray
    }
    
    // Find Material Output node
    const MaterialNode *outputNode = tree.findOutputNode();
    if(!outputNode) {
        std::cerr << "[NodeEval] No Material Output node found\n";
        return Vec3(0.8f, 0.8f, 0.8f);
    }
    
    // Get Surface input
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

// Get emission from node tree
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
    
    // Principled BSDF also has emission
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
