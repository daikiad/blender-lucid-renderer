/**
 * server.hpp - Server mode for persistent renderer process
 * 
 * The server listens on stdin for commands and writes responses to stdout.
 * This avoids the overhead of process startup and scene loading on each frame.
 */

#pragma once

#include "protocol.hpp"
#include "renderer.hpp"  // Include full definitions
#include <string>
#include <memory>
#include <atomic>

/**
 * RenderServer - Main server loop for handling commands
 * 
 * Usage:
 *   RenderServer server;
 *   server.run();  // Blocks until SHUTDOWN command
 */
class RenderServer {
public:
    RenderServer();
    ~RenderServer();
    
    /**
     * Run the server loop.
     * Reads commands from stdin, executes them, writes responses to stdout.
     * Returns when SHUTDOWN command is received or on fatal error.
     */
    void run();

private:
    // Command handlers
    void handleInit(const std::vector<uint8_t>& payload);
    void handleUpdateScene(const std::vector<uint8_t>& payload);
    void handleUpdateCamera(const std::vector<uint8_t>& payload);
    void handleRenderTile(const std::vector<uint8_t>& payload);
    void handleCancel();
    void handleQueryCaps();
    void handleSetBackend(const std::vector<uint8_t>& payload);
    void handleSetAlgorithm(const std::vector<uint8_t>& payload);
    
    // Helper to read payload from stdin
    bool readPayload(uint32_t size, std::vector<uint8_t>& payload);
    
    // Logging (to stderr to not interfere with protocol)
    void log(const std::string& message);
    
    // State
    Scene scene_;
    Camera camera_;
    bool scene_loaded_ = false;
    bool camera_set_ = false;
    
    // Current settings
    std::string backend_ = "cpu";
    std::string algorithm_ = "nee";
    
    // Cancellation flag
    std::atomic<bool> cancel_requested_{false};
    
    // Response writer
    std::unique_ptr<protocol::ResponseWriter> response_;
};
