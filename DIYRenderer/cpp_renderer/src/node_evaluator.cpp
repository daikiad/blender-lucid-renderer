#include "renderer.hpp"
#include "json.hpp"
#include <iostream>
#include <map>
#include <set>
#include <cmath>

using json = nlohmann::json;

// Track already warned node types to avoid log spam
static std::set<std::string> warnedNodeTypes;

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

// ========== JSON Parsing Functions ==========

/**
 * parseSocketValue: Parse JSON value into SocketValue union
 * 
 * Handles multiple types:
 * - Single number → FLOAT
 * - Array[3] → VEC3 (RGB or vector)
 * - Array[4] → VEC4 (RGBA)
 * - Boolean → BOOL
 * - String → STRING
 */
SocketValue parseSocketValue(const json &j) {
    if(j.is_number()) {
        return SocketValue::makeFloat(j.get<float>());
    } else if(j.is_array()) {
        if(j.size() == 3) {
            return SocketValue::makeVec3(j[0], j[1], j[2]);
        } else if(j.size() == 4) {
            float x = j[0], y = j[1], z = j[2], w = j[3];
            return SocketValue::makeVec4(x, y, z, w);
        }
    } else if(j.is_boolean()) {
        SocketValue sv;
        sv.type = SocketValue::BOOL;
        sv.b = j.get<bool>();
        return sv;
    } else if(j.is_string()) {
        SocketValue sv;
        sv.type = SocketValue::STRING;
        sv.s = j.get<std::string>();
        return sv;
    }
    return SocketValue();  // NONE
}

/**
 * parseNodeTree: Construct node graph from JSON
 * 
 * Builds the material node tree by:
 * 1. Creating all MaterialNode objects
 * 2. Parsing input/output sockets for each node
 * 3. Recording socket connections (linked_node, linked_socket)
 * 4. Storing default values for unconnected sockets
 */
NodeTree parseNodeTree(const json &nodeTreeJson) {
    NodeTree tree;
    
    if(!nodeTreeJson.contains("nodes") || !nodeTreeJson.is_object()) {
        return tree;  // Invalid
    }
    
    const auto &nodesArray = nodeTreeJson["nodes"];
    const auto &linksArray = nodeTreeJson.contains("links") ? nodeTreeJson["links"] : json::array();
    
    // Build a map of node connections: (to_node, to_socket) -> (from_node, from_socket)
    std::map<std::pair<std::string, std::string>, std::pair<std::string, std::string>> connections;
    for(const auto &link : linksArray) {
        std::string from_node = link["from_node"];
        std::string from_socket = link["from_socket"];
        std::string to_node = link["to_node"];
        std::string to_socket = link["to_socket"];
        connections[{to_node, to_socket}] = {from_node, from_socket};
    }
    
    // Parse nodes
    for(const auto &nodeJson : nodesArray) {
        MaterialNode node;
        node.name = nodeJson.value("name", "");
        node.type = nodeJson.value("type", "");
        node.label = nodeJson.value("label", "");
        
        // Parse inputs
        if(nodeJson.contains("inputs")) {
            for(const auto &inp : nodeJson["inputs"]) {
                NodeSocket socket;
                socket.name = inp.value("name", "");
                socket.type = inp.value("type", "");
                if(inp.contains("default_value")) {
                    socket.default_value = parseSocketValue(inp["default_value"]);
                }
                
                // Check if this socket is connected
                auto connKey = std::make_pair(node.name, socket.name);
                if(connections.find(connKey) != connections.end()) {
                    socket.is_linked = true;
                    socket.linked_node = connections[connKey].first;
                    socket.linked_socket = connections[connKey].second;
                }
                
                node.inputs.push_back(socket);
            }
        }
        
        // Parse outputs
        if(nodeJson.contains("outputs")) {
            for(const auto &outp : nodeJson["outputs"]) {
                NodeSocket socket;
                socket.name = outp.value("name", "");
                socket.type = outp.value("type", "");
                if(outp.contains("default_value")) {
                    socket.default_value = parseSocketValue(outp["default_value"]);
                }
                node.outputs.push_back(socket);
            }
        }
        
        tree.nodes.push_back(node);
    }
    
    tree.valid = !tree.nodes.empty();
    return tree;
}

// ========== Node Evaluation Functions ==========

/**
 * evaluateNode: Recursively evaluate a node's output socket
 * 
 * @param tree The node tree containing all nodes
 * @param nodeName Name of the node to evaluate
 * @param socketName Name of the output socket to read
 * @param uv UV coordinates for texture lookups
 * @return RGB color value from the socket
 * 
 * This function follows socket connections recursively:
 * 1. Find the requested node by name
 * 2. Check if the input socket is connected
 * 3. If connected: recursively evaluate the linked node
 * 4. If not connected: use the socket's default value
 */
render::ColorRGB evaluateNode(const NodeTree &tree, const std::string &nodeName, const std::string &socketName, const render::Vec2f &uv) {
    const MaterialNode *node = tree.findNode(nodeName);
    if(!node) {
        // Only warn once per missing node name to avoid log spam
        static std::set<std::string> warnedNodes;
        if (warnedNodes.find(nodeName) == warnedNodes.end()) {
            warnedNodes.insert(nodeName);
            std::cerr << "[NodeEval] Node not found: " << nodeName << " (warning once)\n";
        }
        return render::ColorRGB(0.8f, 0.8f, 0.8f);  // Default gray instead of magenta
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
                return render::ColorRGB(0.8f, 0.8f, 0.8f);  // Default gray
            }
            
            if(baseColorSocket->is_linked) {
                // Follow connection
                return evaluateNode(tree, baseColorSocket->linked_node, baseColorSocket->linked_socket, uv);
            } else {
                // Use default value
                if(baseColorSocket->default_value.type == SocketValue::VEC4) {
                    return baseColorSocket->default_value.v4;  // VEC4 uses v4 field (already Color3)
                } else if(baseColorSocket->default_value.type == SocketValue::VEC3) {
                    return render::ColorRGB(baseColorSocket->default_value.v3.x,
                                       baseColorSocket->default_value.v3.y,
                                       baseColorSocket->default_value.v3.z);
                }
                return render::ColorRGB(0.8f, 0.8f, 0.8f);
            }
        } else if(socketName == "Base Color") {
            // Direct query of Base Color
            const NodeSocket *socket = node->findInput("Base Color");
            if(socket && !socket->is_linked) {
                if(socket->default_value.type == SocketValue::VEC4) {
                    return socket->default_value.v4;
                } else if(socket->default_value.type == SocketValue::VEC3) {
                    return render::ColorRGB(socket->default_value.v3.x,
                                       socket->default_value.v3.y,
                                       socket->default_value.v3.z);
                }
            }
            return render::ColorRGB(0.8f, 0.8f, 0.8f);
        }
    } else if(node->type == "ShaderNodeEmission") {
        // ===== Emission Node =====
        // Emission shader: emits light (Color * Strength)
        if(socketName == "Emission") {
            const NodeSocket *colorSocket = node->findInput("Color");
            const NodeSocket *strengthSocket = node->findInput("Strength");
            
            render::ColorRGB color(1.0f, 1.0f, 1.0f);  // Default white
            float strength = 1.0f;          // Default strength
            
            // Evaluate color (can be connected or constant)
            if(colorSocket) {
                if(colorSocket->is_linked) {
                    color = evaluateNode(tree, colorSocket->linked_node, colorSocket->linked_socket, uv);
                } else if(colorSocket->default_value.type == SocketValue::VEC4) {
                    color = colorSocket->default_value.v4;  // RGBA → use v4 field
                } else if(colorSocket->default_value.type == SocketValue::VEC3) {
                    color = render::ColorRGB(colorSocket->default_value.v3.x,
                                        colorSocket->default_value.v3.y,
                                        colorSocket->default_value.v3.z);
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
    } else if(node->type == "ShaderNodeBsdfDiffuse") {
        // ===== Diffuse BSDF Node =====
        // Returns the diffuse color for the surface
        // Similar to Principled BSDF but only handles diffuse component
        if(socketName == "BSDF") {
            // For BSDF output, evaluate the Color input
            const NodeSocket *colorSocket = node->findInput("Color");
            
            if(!colorSocket) {
                return render::ColorRGB(0.8f, 0.8f, 0.8f);  // Default gray
            }
            
            if(colorSocket->is_linked) {
                // Follow connection to linked node
                return evaluateNode(tree, colorSocket->linked_node, colorSocket->linked_socket, uv);
            } else {
                // Use default value
                if(colorSocket->default_value.type == SocketValue::VEC4) {
                    return colorSocket->default_value.v4;  // RGBA → use v4 field (already Color3)
                } else if(colorSocket->default_value.type == SocketValue::VEC3) {
                    return render::ColorRGB(colorSocket->default_value.v3.x,
                                       colorSocket->default_value.v3.y,
                                       colorSocket->default_value.v3.z);
                }
                return render::ColorRGB(0.8f, 0.8f, 0.8f);
            }
        }
    } else if(node->type == "ShaderNodeRGB") {
        // ===== RGB Node =====
        // Simple constant color node (like a color picker in Blender)
        for(const auto &output : node->outputs) {
            if(output.name == "Color" && output.default_value.type != SocketValue::NONE) {
                if(output.default_value.type == SocketValue::VEC4) {
                    return output.default_value.v4;  // RGBA → use v4 field (already Color3)
                } else if(output.default_value.type == SocketValue::VEC3) {
                    return render::ColorRGB(output.default_value.v3.x,
                                       output.default_value.v3.y,
                                       output.default_value.v3.z);
                }
            }
        }
        // Fallback: return white if no valid color found
        return render::ColorRGB(1.0f, 1.0f, 1.0f);
    } else if(node->type == "ShaderNodeMix" || node->type == "ShaderNodeMixRGB") {
        // ===== Mix Node =====
        // Blends two colors using various blend modes (Mix, Add, Multiply, etc.)
        const NodeSocket *facSocket = node->findInput("Fac");
        const NodeSocket *aSocket = node->findInput("A");
        const NodeSocket *bSocket = node->findInput("B");
        
        float fac = 0.5f;
        render::ColorRGB colorA(0, 0, 0), colorB(1, 1, 1);
        
        if(facSocket && !facSocket->is_linked) {
            if(facSocket->default_value.type == SocketValue::FLOAT) {
                fac = facSocket->default_value.f;
            }
        }
        
        if(aSocket) {
            if(aSocket->is_linked) {
                colorA = evaluateNode(tree, aSocket->linked_node, aSocket->linked_socket, uv);
            } else if(aSocket->default_value.type == SocketValue::VEC4) {
                colorA = aSocket->default_value.v4;
            } else if(aSocket->default_value.type == SocketValue::VEC3) {
                colorA = render::ColorRGB(aSocket->default_value.v3.x,
                                     aSocket->default_value.v3.y,
                                     aSocket->default_value.v3.z);
            }
        }
        
        if(bSocket) {
            if(bSocket->is_linked) {
                colorB = evaluateNode(tree, bSocket->linked_node, bSocket->linked_socket, uv);
            } else if(bSocket->default_value.type == SocketValue::VEC4) {
                colorB = bSocket->default_value.v4;
            } else if(bSocket->default_value.type == SocketValue::VEC3) {
                colorB = render::ColorRGB(bSocket->default_value.v3.x,
                                     bSocket->default_value.v3.y,
                                     bSocket->default_value.v3.z);
            }
        }
        
        // Linear interpolation: lerp(A, B, fac) = A * (1 - fac) + B * fac
        return colorA * (1.0f - fac) + colorB * fac;
    } else if(node->type == "ShaderNodeTexChecker") {
        // ===== Checker Texture Node =====
        // Procedural checkerboard pattern using UV coordinates
        const NodeSocket *color1Socket = node->findInput("Color1");
        const NodeSocket *color2Socket = node->findInput("Color2");
        const NodeSocket *scaleSocket = node->findInput("Scale");
        
        render::ColorRGB color1(0.8f, 0.8f, 0.8f);  // Default white
        render::ColorRGB color2(0.2f, 0.2f, 0.2f);  // Default black
        float scale = 5.0f;  // Default scale
        
        if(color1Socket) {
            if(color1Socket->is_linked) {
                color1 = evaluateNode(tree, color1Socket->linked_node, color1Socket->linked_socket, uv);
            } else if(color1Socket->default_value.type == SocketValue::VEC4) {
                color1 = color1Socket->default_value.v4;
            } else if(color1Socket->default_value.type == SocketValue::VEC3) {
                color1 = render::ColorRGB(color1Socket->default_value.v3.x,
                                     color1Socket->default_value.v3.y,
                                     color1Socket->default_value.v3.z);
            }
        }
        
        if(color2Socket) {
            if(color2Socket->is_linked) {
                color2 = evaluateNode(tree, color2Socket->linked_node, color2Socket->linked_socket, uv);
            } else if(color2Socket->default_value.type == SocketValue::VEC4) {
                color2 = color2Socket->default_value.v4;
            } else if(color2Socket->default_value.type == SocketValue::VEC3) {
                color2 = render::ColorRGB(color2Socket->default_value.v3.x,
                                     color2Socket->default_value.v3.y,
                                     color2Socket->default_value.v3.z);
            }
        }
        
        if(scaleSocket && !scaleSocket->is_linked) {
            if(scaleSocket->default_value.type == SocketValue::FLOAT) {
                scale = scaleSocket->default_value.f;
            }
        }
        
        // Checkerboard pattern: alternate colors based on UV coordinates
        float u = uv.x * scale;
        float v = uv.y * scale;
        int checkerU = static_cast<int>(std::floor(u)) % 2;
        int checkerV = static_cast<int>(std::floor(v)) % 2;
        // Handle negative values
        if(checkerU < 0) checkerU += 2;
        if(checkerV < 0) checkerV += 2;
        
        bool isColor1 = (checkerU + checkerV) % 2 == 0;
        return isColor1 ? color1 : color2;
    } else if(node->type == "ShaderNodeOutputMaterial") {
        // ===== Material Output Node =====
        // This is the final output node - traverse to Surface input
        const NodeSocket *surfaceSocket = node->findInput("Surface");
        if(surfaceSocket && surfaceSocket->is_linked) {
            return evaluateNode(tree, surfaceSocket->linked_node, surfaceSocket->linked_socket, uv);
        }
    }
    
    // Unknown node type - return gray (warn once per type)
    if(warnedNodeTypes.find(node->type) == warnedNodeTypes.end()) {
        warnedNodeTypes.insert(node->type);
        std::cerr << "[NodeEval] Unhandled node type: " << node->type << " (further warnings suppressed)\n";
    }
    return render::ColorRGB(0.8f, 0.8f, 0.8f);  // Default gray for unimplemented nodes
}

/**
 * getAlbedoFromNodeTree: Extract base color (diffuse albedo) from node graph
 * 
 * @param tree The material node tree to evaluate
 * @param uv UV coordinates for texture lookups
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
render::ColorRGB getAlbedoFromNodeTree(const NodeTree &tree, const render::Vec2f &uv) {
    static int callCount = 0;
    if(++callCount <= 3) {
        std::cerr << "[NodeEval] getAlbedoFromNodeTree called, valid=" << tree.valid << "\n";
    }
    
    if(!tree.valid) {
        std::cerr << "[NodeEval] Tree not valid, returning gray\n";
        return render::ColorRGB(0.8f, 0.8f, 0.8f);  // Default gray
    }
    
    // Find Material Output node (entry point)
    const MaterialNode *outputNode = tree.findOutputNode();
    if(!outputNode) {
        std::cerr << "[NodeEval] No Material Output node found\n";
        return render::ColorRGB(0.8f, 0.8f, 0.8f);
    }
    
    // Get Surface input (should be connected to a shader like Principled BSDF)
    const NodeSocket *surfaceSocket = outputNode->findInput("Surface");
    if(!surfaceSocket || !surfaceSocket->is_linked) {
        std::cerr << "[NodeEval] Surface socket not connected\n";
        return render::ColorRGB(0.8f, 0.8f, 0.8f);
    }
    
    if(callCount <= 3) {
        std::cerr << "[NodeEval] Evaluating: " << surfaceSocket->linked_node << " / " << surfaceSocket->linked_socket << "\n";
    }
    
    // Evaluate connected BSDF node
    render::ColorRGB result = evaluateNode(tree, surfaceSocket->linked_node, surfaceSocket->linked_socket, uv);
    
    if(callCount <= 3) {
        std::cerr << "[NodeEval] Result: (" << result.r << "," << result.g << "," << result.b << ")\n";
    }
    
    return result;
}

/**
 * getEmissionFromNodeTree: Extract emission (light) from node graph
 * 
 * @param tree The material node tree to evaluate
 * @param uv UV coordinates for texture lookups
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
render::ColorRGB getEmissionFromNodeTree(const NodeTree &tree, const render::Vec2f &uv) {
    if(!tree.valid) {
        return render::ColorRGB(0.0f, 0.0f, 0.0f);  // No emission
    }
    
    // Find Material Output node
    const MaterialNode *outputNode = tree.findOutputNode();
    if(!outputNode) {
        return render::ColorRGB(0.0f, 0.0f, 0.0f);
    }
    
    // Get Surface input
    const NodeSocket *surfaceSocket = outputNode->findInput("Surface");
    if(!surfaceSocket || !surfaceSocket->is_linked) {
        return render::ColorRGB(0.0f, 0.0f, 0.0f);
    }
    
    // Check if connected node is Emission shader
    const MaterialNode *shaderNode = tree.findNode(surfaceSocket->linked_node);
    if(shaderNode && shaderNode->type == "ShaderNodeEmission") {
        return evaluateNode(tree, surfaceSocket->linked_node, "Emission", uv);
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
                render::ColorRGB emissionColor = evaluateNode(tree, emissionSocket->linked_node, emissionSocket->linked_socket, uv);
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
                return render::ColorRGB(emissionSocket->default_value.v3.x,
                                   emissionSocket->default_value.v3.y,
                                   emissionSocket->default_value.v3.z) * strength;
            }
        }
    }
    
    return render::ColorRGB(0.0f, 0.0f, 0.0f);
}

/**
 * getTransmissionFromNodeTree: Extract transmission (glass/transparency) from node graph
 * 
 * @param tree The material node tree to evaluate
 * @param uv UV coordinates (unused for transmission, but kept for API consistency)
 * @return Transmission value (0.0 = opaque, 1.0 = fully transparent)
 * 
 * Algorithm:
 * 1. Find Material Output node
 * 2. Follow its Surface input connection
 * 3. If connected to Principled BSDF, get its Transmission socket
 */
float getTransmissionFromNodeTree(const NodeTree &tree, const render::Vec2f &uv) {
    (void)uv;  // Unused for now
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
 * @param uv UV coordinates (unused for IOR, but kept for API consistency)
 * @return IOR value (1.0 = air, 1.45 = glass, 1.33 = water, 2.42 = diamond)
 * 
 * Algorithm:
 * 1. Find Material Output node
 * 2. Follow its Surface input connection
 * 3. If connected to Principled BSDF, get its IOR socket
 */
float getIORFromNodeTree(const NodeTree &tree, const render::Vec2f &uv) {
    (void)uv;  // Unused for now
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

/**
 * getMetallicFromNodeTree: Extract metallic value from node graph
 * 
 * @param tree The material node tree to evaluate
 * @param uv UV coordinates (unused for metallic, but kept for API consistency)
 * @return Metallic value (0.0 = dielectric, 1.0 = metal)
 */
float getMetallicFromNodeTree(const NodeTree &tree, const render::Vec2f &uv) {
    (void)uv;  // Unused for now
    if(!tree.valid) {
        return 0.0f;  // Default dielectric
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
        const NodeSocket *metallicSocket = shaderNode->findInput("Metallic");
        
        if(metallicSocket && !metallicSocket->is_linked) {
            if(metallicSocket->default_value.type == SocketValue::FLOAT) {
                return metallicSocket->default_value.f;
            }
        }
    }
    
    return 0.0f;  // Default dielectric
}

/**
 * getRoughnessFromNodeTree: Extract roughness value from node graph
 * 
 * @param tree The material node tree to evaluate
 * @param uv UV coordinates (unused for roughness, but kept for API consistency)
 * @return Roughness value (0.0 = smooth/mirror, 1.0 = rough/diffuse)
 */
float getRoughnessFromNodeTree(const NodeTree &tree, const render::Vec2f &uv) {
    (void)uv;  // Unused for now
    if(!tree.valid) {
        return 0.5f;  // Default roughness
    }
    
    // Find Material Output node
    const MaterialNode *outputNode = tree.findOutputNode();
    if(!outputNode) {
        return 0.5f;
    }
    
    // Get Surface input
    const NodeSocket *surfaceSocket = outputNode->findInput("Surface");
    if(!surfaceSocket || !surfaceSocket->is_linked) {
        return 0.5f;
    }
    
    // Check if connected to Principled BSDF
    const MaterialNode *shaderNode = tree.findNode(surfaceSocket->linked_node);
    if(shaderNode && shaderNode->type == "ShaderNodeBsdfPrincipled") {
        const NodeSocket *roughnessSocket = shaderNode->findInput("Roughness");
        
        if(roughnessSocket && !roughnessSocket->is_linked) {
            if(roughnessSocket->default_value.type == SocketValue::FLOAT) {
                return roughnessSocket->default_value.f;
            }
        }
    }
    
    return 0.5f;  // Default roughness
}
