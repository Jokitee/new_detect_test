-- Sub-build integrating migrated sp_vision_25 non-recognition algorithms
-- (11-state EKF, Target, Solver, Aimer, Shooter, TinyMPC Planner) with TAV IFL-Net V2-Slim + Pure-M1.
-- Works both standalone (`cd sp_aim && xmake`) and as an optional target under `new_detect_test` (`xmake build sp_ekf_test`).

local function configure_opencv(target)
    if has_config("use_xrepo") then
        return
    end
    local function try_prefix(prefix)
        if not prefix or #prefix == 0 or not os.isdir(prefix) then
            return false
        end
        local incdir = nil
        for _, cand_inc in ipairs({
            path.join(prefix, "include", "opencv4"),
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
        local libdir = nil
        local links = {}
        for _, cand_lib in ipairs({
            path.join(prefix, "lib"),
            path.join(prefix, "lib64"),
            path.join(prefix, "lib", "x86_64-linux-gnu"),
            path.join(prefix, "lib", "aarch64-linux-gnu")
        }) do
            if os.isdir(cand_lib) then
                if os.isfile(path.join(cand_lib, "libopencv_world.so")) then
                    libdir = cand_lib
                    links = {"opencv_world"}
                    break
                end
                local found = {}
                for _, m in ipairs(mod_order) do
                    if os.isfile(path.join(cand_lib, "lib" .. m .. ".so")) then
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
        if #home > 0 and user_dir:sub(1, 2) == "~/" then
            user_dir = path.join(home, user_dir:sub(3))
        end
        table.insert(candidates, user_dir)
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

    import("lib.detect.find_package")
    for _, name in ipairs({"pkgconfig::opencv4", "pkgconfig::opencv", "cmake::OpenCV"}) do
        local pkg = find_package(name)
        if pkg then
            if pkg.includedirs then target:add("includedirs", pkg.includedirs) end
            if pkg.sysincludedirs then target:add("includedirs", pkg.sysincludedirs) end
            if pkg.linkdirs then
                target:add("linkdirs", pkg.linkdirs)
                target:add("rpathdirs", pkg.linkdirs)
            end
            if pkg.links then target:add("links", pkg.links) end
            return
        end
    end
    raise("OpenCV not found for sp_ekf_test.")
end

local function configure_sp_deps(target)
    import("lib.detect.find_package")
    local function try_pkg(names, fallback_incs, fallback_links)
        for _, n in ipairs(names) do
            local pkg = find_package(n)
            if pkg then
                if pkg.includedirs then target:add("includedirs", pkg.includedirs) end
                if pkg.sysincludedirs then target:add("includedirs", pkg.sysincludedirs) end
                if pkg.defines then target:add("defines", pkg.defines) end
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

    try_pkg({"pkgconfig::eigen3", "cmake::Eigen3"}, {"/usr/include/eigen3", "/usr/local/include/eigen3"}, {})
    try_pkg({"pkgconfig::yaml-cpp", "cmake::yaml-cpp"}, {}, {"yaml-cpp"})
    try_pkg({"pkgconfig::fmt", "cmake::fmt"}, {}, {"fmt"})
    if not try_pkg({"pkgconfig::spdlog", "cmake::spdlog"}, {}, {"spdlog"}) then
        -- Ubuntu apt libspdlog-dev defaults to compiled lib against external fmt
        target:add("defines", "SPDLOG_FMT_EXTERNAL", "SPDLOG_COMPILED_LIB")
    end
    try_pkg({"pkgconfig::nlohmann_json", "cmake::nlohmann_json"}, {}, {})
end

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
        if not has_config("use_xrepo") then
            configure_opencv(target)
        end
        configure_sp_deps(target)
    end)

    if has_config("with_mvsdk") then
        add_defines("TAV_WITH_MVSDK=1")
        add_links("MVSDK")
        local sdkdir = get_config("mvsdk_dir")
        if sdkdir and #sdkdir > 0 then
            add_includedirs(path.join(sdkdir, "include"))
            add_linkdirs(path.join(sdkdir, "lib"))
        end
    else
        add_defines("TAV_WITH_MVSDK=0")
    end

    set_rundir(root_dir)
    if is_plat("linux") then
        add_syslinks("pthread")
        add_ldflags("-rdynamic")
    end
