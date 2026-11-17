set_project("new_detect_test")
set_version("1.1.0")
set_languages("c++17")
set_optimize("fastest")
set_warnings("all")

add_rules("mode.debug", "mode.release")

option("opencv_ver")
    set_default("4.13.x")
    set_showmenu(true)
    set_description("OpenCV version constraint (e.g. 4.13.x, 4.13.0, or 4.x)")
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

add_requires("opencv " .. (get_config("opencv_ver") or "4.13.x"), {configs = {shared = true}})

target("new_detect_test")
    set_kind("binary")
    add_includedirs("include")
    add_files("src/*.cpp")
    add_packages("opencv")
    add_options("with_mvsdk", "mvsdk_dir")
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
