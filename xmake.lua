set_project("new_detect_test")
set_version("1.1.0")
set_languages("c++17")
set_optimize("fastest")
set_warnings("all")

add_rules("mode.debug", "mode.release")

option("use_xrepo")
    set_default(false)
    set_showmenu(true)
    set_description("Download OpenCV via xmake-repo instead of using local/system OpenCV")
option_end()

option("opencv_dir")
    set_default("")
    set_showmenu(true)
    set_description("Optional custom OpenCV install prefix (containing include/opencv4 and lib/)")
option_end()

option("opencv_ver")
    set_default("4.13.x")
    set_showmenu(true)
    set_description("OpenCV version constraint when --use_xrepo=true")
option_end()

option("with_mvsdk")
    set_default(true)
    set_showmenu(true)
    set_description("Link MindVision industrial camera SDK (MVSDK)")
option_end()

option("mvsdk_dir")
    set_default("")
    set_showmenu(true)
    set_description("Optional custom MindVision SDK root (containing include/ and lib/)")
option_end()

if has_config("use_xrepo") then
    add_requires("opencv " .. (get_config("opencv_ver") or "4.13.x"), {configs = {shared = true}})
end

target("new_detect_test")
    set_kind("binary")
    add_includedirs("include")
    add_files("src/*.cpp")
    add_options("use_xrepo", "opencv_dir", "with_mvsdk", "mvsdk_dir")

    if has_config("use_xrepo") then
        add_packages("opencv")
    else
        on_load(function (target)
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

            raise("Local OpenCV not found. Pass `xmake f --opencv_dir=/path/to/opencv_install` or `xmake f --use_xrepo=true`.")
        end)
    end

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
    set_rundir("$(projectdir)")
    if is_plat("linux") then
        add_ldflags("-rdynamic")
    end
