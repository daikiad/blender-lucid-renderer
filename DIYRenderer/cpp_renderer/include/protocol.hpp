/**
 * protocol.hpp - Binary protocol for Python <-> C++ communication
 * 
 * Protocol format:
 *   Command:  [Magic 4B][CmdType 4B][PayloadSize 4B][Payload...]
 *   Response: [Magic 4B][RespType 4B][Status 4B][PayloadSize 4B][Payload...]
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <iostream>

namespace protocol {

// Protocol constants
constexpr uint32_t MAGIC = 0x44495952;  // "DIYR"
constexpr size_t COMMAND_HEADER_SIZE = 12;
constexpr size_t RESPONSE_HEADER_SIZE = 16;

// Command types (Python -> C++)
enum class CommandType : uint32_t {
    INIT = 0x01,
    UPDATE_SCENE = 0x02,
    UPDATE_CAMERA = 0x03,
    RENDER_TILE = 0x04,
    CANCEL = 0x05,
    QUERY_CAPS = 0x06,
    SET_BACKEND = 0x07,
    SET_ALGORITHM = 0x08,
    SHUTDOWN = 0xFF
};

// Response types (C++ -> Python)
enum class ResponseType : uint32_t {
    ACK = 0x81,
    PIXELS = 0x82,
    PROGRESS = 0x83,
    CAPABILITIES = 0x84,
    ERROR = 0x85
};

// Status codes
enum class StatusCode : uint32_t {
    OK = 0,
    ERROR_UNKNOWN = 1,
    ERROR_INVALID_COMMAND = 2,
    ERROR_SCENE_NOT_LOADED = 3,
    ERROR_RENDER_FAILED = 4,
    ERROR_CANCELLED = 5
};

// Camera parameters (40 bytes)
struct CameraParams {
    float pos_x, pos_y, pos_z;
    float dir_x, dir_y, dir_z;
    float up_x, up_y, up_z;
    float fov;
    
    static CameraParams fromBytes(const uint8_t* data) {
        CameraParams cam;
        const float* f = reinterpret_cast<const float*>(data);
        cam.pos_x = f[0]; cam.pos_y = f[1]; cam.pos_z = f[2];
        cam.dir_x = f[3]; cam.dir_y = f[4]; cam.dir_z = f[5];
        cam.up_x = f[6]; cam.up_y = f[7]; cam.up_z = f[8];
        cam.fov = f[9];
        return cam;
    }
};

// Render tile parameters (36 bytes)
struct RenderTileParams {
    uint32_t tile_x, tile_y, tile_w, tile_h;
    uint32_t full_w, full_h;
    uint32_t samples;
    uint32_t sample_offset;
    uint32_t max_depth;
    
    static RenderTileParams fromBytes(const uint8_t* data) {
        RenderTileParams p;
        const uint32_t* u = reinterpret_cast<const uint32_t*>(data);
        p.tile_x = u[0]; p.tile_y = u[1]; p.tile_w = u[2]; p.tile_h = u[3];
        p.full_w = u[4]; p.full_h = u[5];
        p.samples = u[6]; p.sample_offset = u[7]; p.max_depth = u[8];
        return p;
    }
};

// Command header
struct CommandHeader {
    uint32_t magic;
    CommandType type;
    uint32_t payload_size;
    
    bool isValid() const { return magic == MAGIC; }
    
    static bool read(std::istream& in, CommandHeader& header) {
        in.read(reinterpret_cast<char*>(&header.magic), 4);
        uint32_t type_raw;
        in.read(reinterpret_cast<char*>(&type_raw), 4);
        header.type = static_cast<CommandType>(type_raw);
        in.read(reinterpret_cast<char*>(&header.payload_size), 4);
        return in.good() && header.isValid();
    }
};

// Response writer
class ResponseWriter {
public:
    explicit ResponseWriter(std::ostream& out) : out_(out) {}
    
    void writeAck(StatusCode status = StatusCode::OK) {
        writeHeader(ResponseType::ACK, status, 0);
        out_.flush();
    }
    
    void writePixels(const std::vector<float>& pixels, int width, int height) {
        size_t payload_size = width * height * 4 * sizeof(float);
        writeHeader(ResponseType::PIXELS, StatusCode::OK, payload_size);
        out_.write(reinterpret_cast<const char*>(pixels.data()), payload_size);
        out_.flush();
    }
    
    void writeProgress(float progress) {
        writeHeader(ResponseType::PROGRESS, StatusCode::OK, sizeof(float));
        out_.write(reinterpret_cast<const char*>(&progress), sizeof(float));
        out_.flush();
    }
    
    void writeCapabilities(const std::vector<std::string>& backends,
                           const std::vector<std::string>& algorithms) {
        std::vector<uint8_t> payload;
        
        auto writeStringList = [&payload](const std::vector<std::string>& list) {
            uint32_t count = static_cast<uint32_t>(list.size());
            payload.insert(payload.end(), 
                reinterpret_cast<uint8_t*>(&count),
                reinterpret_cast<uint8_t*>(&count) + 4);
            
            for (const auto& s : list) {
                uint32_t len = static_cast<uint32_t>(s.size());
                payload.insert(payload.end(),
                    reinterpret_cast<uint8_t*>(&len),
                    reinterpret_cast<uint8_t*>(&len) + 4);
                payload.insert(payload.end(), s.begin(), s.end());
            }
        };
        
        writeStringList(backends);
        writeStringList(algorithms);
        
        writeHeader(ResponseType::CAPABILITIES, StatusCode::OK, payload.size());
        out_.write(reinterpret_cast<const char*>(payload.data()), payload.size());
        out_.flush();
    }
    
    void writeError(StatusCode status, const std::string& message) {
        writeHeader(ResponseType::ERROR, status, message.size());
        out_.write(message.data(), message.size());
        out_.flush();
    }

private:
    void writeHeader(ResponseType type, StatusCode status, size_t payload_size) {
        uint32_t magic = MAGIC;
        uint32_t type_raw = static_cast<uint32_t>(type);
        uint32_t status_raw = static_cast<uint32_t>(status);
        uint32_t size = static_cast<uint32_t>(payload_size);
        
        out_.write(reinterpret_cast<const char*>(&magic), 4);
        out_.write(reinterpret_cast<const char*>(&type_raw), 4);
        out_.write(reinterpret_cast<const char*>(&status_raw), 4);
        out_.write(reinterpret_cast<const char*>(&size), 4);
    }
    
    std::ostream& out_;
};

// Payload reader helper
class PayloadReader {
public:
    PayloadReader(const std::vector<uint8_t>& data) : data_(data), offset_(0) {}
    
    std::string readString() {
        if (offset_ + 4 > data_.size()) return "";
        uint32_t len = *reinterpret_cast<const uint32_t*>(&data_[offset_]);
        offset_ += 4;
        if (offset_ + len > data_.size()) return "";
        std::string s(data_.begin() + offset_, data_.begin() + offset_ + len);
        offset_ += len;
        return s;
    }
    
    const uint8_t* data() const { return data_.data() + offset_; }
    size_t remaining() const { return data_.size() - offset_; }

private:
    const std::vector<uint8_t>& data_;
    size_t offset_;
};

}  // namespace protocol
