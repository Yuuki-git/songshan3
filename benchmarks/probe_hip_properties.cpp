#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime_api.h>
#include <hip/hip_deprecated.h>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>

int main() {
    void* library = dlopen("/opt/dtk/lib/libamdhip64.so", RTLD_NOW | RTLD_LOCAL);
    if (!library) { std::fprintf(stderr, "%s\n", dlerror()); return 1; }
    using GetProps = hipError_t (*)(hipDeviceProp_tR0000*, int);
    auto get = reinterpret_cast<GetProps>(dlsym(library, "hipGetDeviceProperties"));
    if (!get) { std::fprintf(stderr, "%s\n", dlerror()); return 1; }
    // The large aligned allocation also makes the inspection safe if a vendor
    // appends fields beyond the old property structure.
    alignas(std::max_align_t) unsigned char buffer[8192]{};
    std::memset(buffer, 0xa5, sizeof(buffer));
    auto* properties = reinterpret_cast<hipDeviceProp_tR0000*>(buffer);
    auto status = get(properties, 0);
    if (status != hipSuccess) { std::printf("HIP status %d\n", status); return 1; }
    for (std::size_t i = sizeof(*properties); i < sizeof(buffer); ++i) {
        if (buffer[i] != 0xa5) {
            std::fprintf(stderr, "Vendor wrote beyond the legacy structure at offset %zu\n", i);
            return 1;
        }
    }
    if (!std::memchr(properties->gcnArchName, 0, sizeof(properties->gcnArchName))) {
        std::fprintf(stderr, "Unterminated architecture field\n"); return 1;
    }
    std::printf("{\"legacy_size\":%zu,\"arch_offset\":%zu,\"name\":\"%s\","
                "\"arch\":\"%s\",\"warp_size\":%d,\"sm_count\":%d,"
                "\"shared_memory\":%zu,\"total_memory\":%zu}\n",
                sizeof(*properties), offsetof(hipDeviceProp_tR0000, gcnArchName),
                properties->name, properties->gcnArchName, properties->warpSize,
                properties->multiProcessorCount, properties->sharedMemPerBlock,
                properties->totalGlobalMem);
    return std::strncmp(properties->gcnArchName, "gfx906", 6) != 0;
}
