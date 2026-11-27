-- Sub-build integrating migrated sp_vision_25 non-recognition algorithms
-- (11-state EKF, Target, Solver, Aimer, Shooter, TinyMPC Planner) with TAV IFL-Net V2-Slim + Pure-M1.
-- Works both standalone (`cd sp_aim && xmake`) and as an optional target under `new_detect_test` (`xmake build sp_ekf_test`).

local sp_dir = os.scriptdir()
local root_dir = path.directory(sp_dir)
local is_standalone = (os.projectdir() == sp_dir)

if is_standalone then
    set_project("sp_ekf_test")
    set_languages("c++17")
    set_optimize("fastest")
    set_warnings("all")
    add_rules("mode.debug", "mode.release")

    option("use_xrepo")
        set_default(false)
        set_showmenu(true)
    option_end()
    option("opencv_dir")
        set_default("")
        set_showmenu(true)
    option_end()
    option("with_mvsdk")
        set_default(true)
        set_showmenu(true)
    option_end()
    option("mvsdk_dir")
        set_default("")
        set_showmenu(true)
    option_end()
end

target("sp_ekf_test")
    set_kind("binary")
    set_default(is_standalone)
    add_options("use_xrepo", "opencv_dir", "with_mvsdk", "mvsdk_dir")

    add_defines(
        'SP_AIM_SOURCE_DIR="' .. sp_dir:gsub("\\", "/") .. '"',
        'NEW_DETECT_ROOT_DIR="' .. root_dir:gsub("\\", "/") .. '"'
    )

    add_includedirs(
        path.join(root_dir, "include"),
        path.join(sp_dir, "include"),
        path.join(sp_dir, "sp_port"),
        path.join(sp_dir, "sp_port", "tasks", "auto_aim", "planner")
    )

    -- Reuse TAV detector, Pure-M1 solver & hardened MindVision camera (NO original recognition code from sp_vision_25)
    add_files(
        path.join(root_dir, "src", "gpu_accel.cpp"),
        path.join(root_dir, "src", "yolo_detector.cpp"),
        path.join(root_dir, "src", "sparse_edge_pose.cpp"),
        path.join(root_dir, "src", "mv_camera.cpp"),
        path.join(sp_dir, "src", "pure_m1_bridge.cpp"),
        path.join(sp_dir, "src", "m1_tracker.cpp"),
        path.join(sp_dir, "src", "sp_ekf_test_main.cpp")
    )

    -- Migrated sp_vision_25 non-recognition algorithms (EKF, Math, Trajectory, Solver, Target, Aimer, Shooter, TinyMPC Planner)
    add_files(
        path.join(sp_dir, "sp_port", "tools", "exiter.cpp"),
        path.join(sp_dir, "sp_port", "tools", "extended_kalman_filter.cpp"),
        path.join(sp_dir, "sp_port", "tools", "img_tools.cpp"),
        path.join(sp_dir, "sp_port", "tools", "logger.cpp"),
        path.join(sp_dir, "sp_port", "tools", "math_tools.cpp"),
        path.join(sp_dir, "sp_port", "tools", "plotter.cpp"),
        path.join(sp_dir, "sp_port", "tools", "recorder.cpp"),
        path.join(sp_dir, "sp_port", "tools", "trajectory.cpp"),
        path.join(sp_dir, "sp_port", "tasks", "auto_aim", "armor.cpp"),
        path.join(sp_dir, "sp_port", "tasks", "auto_aim", "solver.cpp"),
        path.join(sp_dir, "sp_port", "tasks", "auto_aim", "target.cpp"),
        path.join(sp_dir, "sp_port", "tasks", "auto_aim", "aimer.cpp"),
        path.join(sp_dir, "sp_port", "tasks", "auto_aim", "shooter.cpp"),
        path.join(sp_dir, "sp_port", "tasks", "auto_aim", "voter.cpp"),
        path.join(sp_dir, "sp_port", "tasks", "auto_aim", "planner", "planner.cpp"),
        path.join(sp_dir, "sp_port", "tasks", "auto_aim", "planner", "tinympc", "*.cpp")
    )

    if has_config("use_xrepo") then
        add_packages("opencv")
    end

    on_load(function (target)
        import("lib.detect.find_package")
        local arch = (target.arch and target:arch()) or os.arch() or ""
        local is_arm64 = arch:find("arm64") or arch:find("aarch64")

        local function expand_home(p)
            local home = os.getenv("HOME") or ""
            if #home > 0 and p:sub(1, 2) == "~/" then
                return path.join(home, p:sub(3))
            end
            return p
        end

        local function configure_opencv()
            local function try_prefix(prefix)
                if not prefix or #prefix == 0 or not os.isdir(prefix) then
                    return false
                end
                local incdir = nil
                for _, cand_inc in ipairs({
                    path.join(prefix, "include", "opencv4"),
                    path.join(prefix, "include", "aarch64-linux-gnu", "opencv4"),
                    path.join(prefix, "include", "x86_64-linux-gnu", "opencv4"),
                    path.join(prefix, "include")
                }) do
                    if os.isfile(path.join(cand_inc, "opencv2", "opencv.hpp")) then
                        incdir = cand_inc
                        break
                    end
                end
                if not incdir then
                    return false
                end

                local mod_order = {
                    "opencv_dnn", "opencv_calib3d", "opencv_highgui", "opencv_videoio",
                    "opencv_imgcodecs", "opencv_features2d", "opencv_flann",
                    "opencv_imgproc", "opencv_core"
                }
                local lib_candidates = is_arm64 and {
                    path.join(prefix, "lib"),
                    path.join(prefix, "lib", "aarch64-linux-gnu"),
                    path.join(prefix, "lib64"),
                    path.join(prefix, "lib", "x86_64-linux-gnu")
                } or {
                    path.join(prefix, "lib"),
                    path.join(prefix, "lib64"),
                    path.join(prefix, "lib", "x86_64-linux-gnu"),
                    path.join(prefix, "lib", "aarch64-linux-gnu")
                }
                local libdir = nil
                local links = {}
                for _, cand_lib in ipairs(lib_candidates) do
                    if os.isdir(cand_lib) then
                        if os.isfile(path.join(cand_lib, "libopencv_world.so")) or
                           os.isfile(path.join(cand_lib, "libopencv_world.a")) then
                            libdir = cand_lib
                            links = {"opencv_world"}
                            break
                        end
                        local found = {}
                        for _, m in ipairs(mod_order) do
                            if os.isfile(path.join(cand_lib, "lib" .. m .. ".so")) or
                               os.isfile(path.join(cand_lib, "lib" .. m .. ".a")) then
                                table.insert(found, m)
                            end
                        end
                        if #found > 0 then
                            libdir = cand_lib
                            links = found
                            break
                        end
                    end
                end
                if not libdir then
                    return false
                end

                target:add("includedirs", incdir)
                target:add("linkdirs", libdir)
                target:add("rpathdirs", libdir)
                target:add("links", links)
                return true
            end

            local candidates = {}
            local home = os.getenv("HOME") or ""
            local user_dir = get_config("opencv_dir")
            if user_dir and #user_dir > 0 then
                table.insert(candidates, expand_home(user_dir))
            end
            for _, env_name in ipairs({"OpenCV_DIR", "OPENCV_DIR"}) do
                local ev = os.getenv(env_name)
                if ev and #ev > 0 then
                    table.insert(candidates, ev)
                    local p2 = path.directory(path.directory(ev))
                    table.insert(candidates, p2)
                    table.insert(candidates, path.directory(p2))
                end
            end
            if #home > 0 then
                table.insert(candidates, path.join(home, "install"))
                table.insert(candidates, path.join(home, "install", "opencv4"))
                table.insert(candidates, path.join(home, "install", "opencv-4.13.0"))
                table.insert(candidates, path.join(home, "opencv", "install"))
                table.insert(candidates, path.join(home, "opencv-4.13.0", "install"))
            end
            table.insert(candidates, "/usr/local")
            table.insert(candidates, "/usr")

            for _, p in ipairs(candidates) do
                if try_prefix(p) then
                    return
                end
            end

            for _, name in ipairs({"pkgconfig::opencv4", "pkgconfig::opencv", "cmake::OpenCV"}) do
                local pkg = find_package(name)
                if pkg then
                    if pkg.includedirs then target:add("includedirs", pkg.includedirs) end
                    if pkg.sysincludedirs then target:add("includedirs", pkg.sysincludedirs) end
                    if pkg.defines then target:add("defines", pkg.defines) end
                    if pkg.cxflags then target:add("cxflags", pkg.cxflags) end
                    if pkg.cxxflags then target:add("cxxflags", pkg.cxxflags) end
                    if pkg.linkdirs then
                        target:add("linkdirs", pkg.linkdirs)
                        target:add("rpathdirs", pkg.linkdirs)
                    end
                    if pkg.links then target:add("links", pkg.links) end
                    return
                end
            end
            raise("OpenCV not found for sp_ekf_test (" .. arch .. ").")
        end

        local function configure_mvsdk()
            if not has_config("with_mvsdk") then
                target:add("defines", "TAV_WITH_MVSDK=0")
                return
            end

            local sdk_roots = {}
            local home = os.getenv("HOME") or ""
            local user_sdk = get_config("mvsdk_dir")
            if user_sdk and #user_sdk > 0 then
                table.insert(sdk_roots, expand_home(user_sdk))
            end
            for _, env_name in ipairs({"MVSDK_DIR", "MINDVISION_DIR"}) do
                local ev = os.getenv(env_name)
                if ev and #ev > 0 then
                    table.insert(sdk_roots, expand_home(ev))
                end
            end
            if #home > 0 then
                table.insert(sdk_roots, path.join(home, "MVSDK"))
                table.insert(sdk_roots, path.join(home, "mvsdk"))
                for _, d in ipairs(os.dirs(path.join(home, "linuxSDK*")) or {}) do
                    table.insert(sdk_roots, d)
                end
            end
            table.insert(sdk_roots, "/opt/MindVision")
            table.insert(sdk_roots, "/usr/local")
            table.insert(sdk_roots, "/usr")

            local lib_subdirs = is_arm64
                and {"lib/arm64", "lib/aarch64", "lib/aarch64-linux-gnu", "lib", "arm64", "aarch64"}
                or  {"lib/x64", "lib/x86_64", "lib/x86_64-linux-gnu", "lib64", "lib", "x64", "x86_64"}

            local found_inc, found_lib = nil, nil
            for _, root in ipairs(sdk_roots) do
                if os.isdir(root) then
                    local inc = nil
                    for _, ci in ipairs({path.join(root, "include"), root}) do
                        if os.isfile(path.join(ci, "CameraApi.h")) then
                            inc = ci
                            break
                        end
                    end
                    local libd = nil
                    for _, sub in ipairs(lib_subdirs) do
                        local cl = path.join(root, sub)
                        if os.isfile(path.join(cl, "libMVSDK.so")) or
                           os.isfile(path.join(cl, "libMVSDK.a")) then
                            libd = cl
                            break
                        end
                    end
                    if inc and not found_inc then found_inc = inc end
                    if libd and not found_lib then found_lib = libd end
                    if inc and libd then
                        found_inc, found_lib = inc, libd
                        break
                    end
                end
            end

            if not found_inc or not found_lib then
                local expected_sub = is_arm64 and "lib/arm64 (Raspberry Pi 5)" or "lib/x64 (Intel NUC)"
                raise("MindVision SDK (CameraApi.h / libMVSDK.so for " .. arch .. ", " .. expected_sub .. ") not found for sp_ekf_test.\n"
                   .. "  - Install MindVision Linux SDK or pass `--mvsdk_dir=/path/to/linuxSDK` or `--with_mvsdk=false`.")
            end

            target:add("defines", "TAV_WITH_MVSDK=1")
            target:add("includedirs", found_inc)
            target:add("linkdirs", found_lib)
            target:add("rpathdirs", found_lib)
            target:add("links", "MVSDK")
        end

        local function try_pkg(names, fallback_incs, fallback_links)
            for _, n in ipairs(names) do
                local pkg = find_package(n)
                if pkg then
                    if pkg.includedirs then target:add("includedirs", pkg.includedirs) end
                    if pkg.sysincludedirs then target:add("includedirs", pkg.sysincludedirs) end
                    if pkg.defines then target:add("defines", pkg.defines) end
                    if pkg.cxflags then target:add("cxflags", pkg.cxflags) end
                    if pkg.cxxflags then target:add("cxxflags", pkg.cxxflags) end
                    if pkg.linkdirs then target:add("linkdirs", pkg.linkdirs) end
                    if pkg.links then target:add("links", pkg.links) end
                    return true
                end
            end
            for _, inc in ipairs(fallback_incs or {}) do
                if os.isdir(inc) then
                    target:add("includedirs", inc)
                end
            end
            for _, l in ipairs(fallback_links or {}) do
                target:add("links", l)
            end
            return false
        end

        if not has_config("use_xrepo") then
            configure_opencv()
        end
        configure_mvsdk()
        try_pkg({"pkgconfig::eigen3", "cmake::Eigen3"}, {"/usr/include/eigen3", "/usr/local/include/eigen3"}, {})
        try_pkg({"pkgconfig::yaml-cpp", "cmake::yaml-cpp"}, {}, {"yaml-cpp"})
        try_pkg({"pkgconfig::fmt", "cmake::fmt"}, {}, {"fmt"})
        if not try_pkg({"pkgconfig::spdlog", "cmake::spdlog"}, {}, {"spdlog"}) then
            target:add("defines", "SPDLOG_FMT_EXTERNAL", "SPDLOG_COMPILED_LIB")
        end
        try_pkg({"pkgconfig::nlohmann_json", "cmake::nlohmann_json"}, {}, {})
    end)

    set_rundir(root_dir)
    if is_plat("linux") then
        add_syslinks("pthread", "dl")
        add_ldflags("-rdynamic")
    end
