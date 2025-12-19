-- ============================================================
-- REAPER BATCH MIDI RENDERER
-- ============================================================
-- Renders MIDI files to audio using a specified VST instrument
-- 
-- USAGE:
-- 1. Configure settings below
-- 2. Actions > Show action list > Load ReaScript
-- 3. Select this file and run
--
-- REQUIREMENTS:
-- - Reaper DAW
-- - VST instrument installed (e.g., Equator2, Kontakt, etc.)
-- ============================================================

-- ============================================================
-- CONFIGURATION
-- ============================================================

-- Input/Output paths (use forward slashes or double backslashes)
MIDI_FOLDER = "path/to/midi/folder"
OUTPUT_FOLDER = "path/to/output/folder"

-- VST instrument settings
VST_NAME = "VST3i: Equator2 (ROLI)"  -- VST identifier (check in Reaper FX browser)
VST_PRESET = "Preset Name"           -- Preset name (optional, leave empty if not needed)

-- Render settings
OUTPUT_FORMAT = "wav"                 -- Format: wav, flac, mp3
RENDER_SAMPLE_RATE = 48000           -- Sample rate in Hz
RENDER_BIT_DEPTH = 24                -- Bit depth: 16, 24, 32
RENDER_NORMALIZE = false             -- Normalize output audio
RENDER_CHANNELS = 2                  -- 1 = mono, 2 = stereo

-- Performance settings
WAIT_AFTER_PRESET_LOAD = 0.5         -- Seconds to wait for plugin initialization

-- ============================================================
-- HELPER FUNCTIONS (DO NOT EDIT)
-- ============================================================

function log(text)
    reaper.ShowConsoleMsg(text .. "\n")
end

function fix_path(path)
    if string.sub(path, -1) ~= "/" and string.sub(path, -1) ~= "\\" then
        return path .. "/"
    end
    return path:gsub("\\", "/")
end

function get_midi_files(folder)
    local files = {}
    local i = 0
    local file = reaper.EnumerateFiles(folder, i)
    
    while file do
        if file:match("%.mid$") or file:match("%.midi$") then
            table.insert(files, file)
        end
        i = i + 1
        file = reaper.EnumerateFiles(folder, i)
    end
    
    return files
end

function wait(seconds)
    local start_time = reaper.time_precise()
    while (reaper.time_precise() - start_time) < seconds do
        -- Busy wait for plugin initialization
    end
end

function setup_track()
    log("Setting up track with VST instrument...")
    
    -- Create new track
    reaper.Main_OnCommand(40001, 0)  -- Insert new track
    local track = reaper.GetTrack(0, 0)
    
    -- Add VST instrument
    local fx_id = reaper.TrackFX_AddByName(track, VST_NAME, false, -1)
    
    if fx_id == -1 then
        reaper.MB(
            "Cannot find VST instrument.\n\n" ..
            "Check VST_NAME setting:\n" .. VST_NAME,
            "ERROR", 0
        )
        return nil, nil
    end
    
    log("  ✓ VST loaded: " .. VST_NAME)
    
    -- Load preset if specified
    if VST_PRESET ~= "" then
        local preset_ok = reaper.TrackFX_SetPreset(track, fx_id, VST_PRESET)
        if not preset_ok then
            log("  ⚠ Could not load preset: " .. VST_PRESET)
        else
            log("  ✓ Preset loaded: " .. VST_PRESET)
        end
    end
    
    -- Wait for plugin initialization
    wait(WAIT_AFTER_PRESET_LOAD)
    
    return track, fx_id
end

function setup_render_settings(output_folder, output_filename)
    -- Sample rate
    reaper.GetSetProjectInfo(0, "RENDER_SRATE", RENDER_SAMPLE_RATE, true)
    
    -- Bit depth
    reaper.GetSetProjectInfo(0, "RENDER_BITDEPTH", RENDER_BIT_DEPTH, true)
    
    -- Normalize
    reaper.GetSetProjectInfo_String(
        0, "RENDER_NORMALIZE",
        RENDER_NORMALIZE and "1" or "0",
        true
    )
    
    -- Output directory
    reaper.GetSetProjectInfo_String(0, "RENDER_FILE", output_folder, true)
    
    -- Output filename
    reaper.GetSetProjectInfo_String(0, "RENDER_PATTERN", output_filename, true)
    
    -- Output format
    local format_map = {
        wav = "evaw",
        mp3 = "l3pm",
        flac = "CALF"
    }
    local format_str = format_map[OUTPUT_FORMAT] or "evaw"
    reaper.GetSetProjectInfo_String(0, "RENDER_FORMAT", format_str, true)
    
    -- Channels (mono/stereo)
    reaper.GetSetProjectInfo(0, "RENDER_CHANNELS", RENDER_CHANNELS, true)
end

function render_file(midi_file)
    -- Clear project
    reaper.Main_OnCommand(40006, 0)  -- Delete all items
    
    -- Import MIDI file
    local fullpath = MIDI_FOLDER .. midi_file
    reaper.InsertMedia(fullpath, 0)
    
    -- Generate output filename
    local basename = midi_file:gsub("%.midi?$", "")
    local output_filename = basename .. "." .. OUTPUT_FORMAT
    
    log("  → " .. output_filename)
    
    -- Configure render settings
    setup_render_settings(OUTPUT_FOLDER, output_filename)
    
    -- Select all items and set time selection
    reaper.Main_OnCommand(40290, 0)  -- Select all items
    reaper.Main_OnCommand(40225, 0)  -- Set time selection to items
    
    -- Render (blocking operation)
    reaper.Main_OnCommand(42230, 0)  -- Render using last settings
end

-- ============================================================
-- MAIN SCRIPT
-- ============================================================

function main()
    log("=" .. string.rep("=", 58))
    log("REAPER BATCH MIDI RENDERER")
    log("=" .. string.rep("=", 58))
    log("")
    
    -- Normalize paths
    MIDI_FOLDER = fix_path(MIDI_FOLDER)
    OUTPUT_FOLDER = fix_path(OUTPUT_FOLDER)
    
    -- Display configuration
    log("Configuration:")
    log("  MIDI folder:    " .. MIDI_FOLDER)
    log("  Output folder:  " .. OUTPUT_FOLDER)
    log("  VST:            " .. VST_NAME)
    if VST_PRESET ~= "" then
        log("  Preset:         " .. VST_PRESET)
    end
    log("  Format:         " .. OUTPUT_FORMAT .. " @ " .. RENDER_SAMPLE_RATE .. "Hz, " .. RENDER_BIT_DEPTH .. "-bit")
    log("")
    
    -- Find MIDI files
    local files = get_midi_files(MIDI_FOLDER)
    
    if #files == 0 then
        reaper.MB(
            "No MIDI files found in:\n" .. MIDI_FOLDER,
            "Error", 0
        )
        log("ERROR: No MIDI files found!")
        return
    end
    
    log("Found " .. #files .. " MIDI files")
    log("")
    
    -- Confirm before processing
    local estimated_time = math.ceil(#files * 0.5)
    local confirm = reaper.MB(
        "Ready to render " .. #files .. " MIDI files.\n\n" ..
        "Estimated time: ~" .. estimated_time .. " minutes\n\n" ..
        "Output: " .. OUTPUT_FOLDER .. "\n\n" ..
        "Continue?",
        "Confirm Batch Render",
        4  -- Yes/No buttons
    )
    
    if confirm ~= 6 then  -- 6 = Yes
        log("Cancelled by user")
        return
    end
    
    -- Setup track with VST
    local track, fx_id = setup_track()
    if not track then
        return
    end
    log("")
    
    -- Process each file
    local start_time = reaper.time_precise()
    log("Rendering files...")
    log("")
    
    for i, filename in ipairs(files) do
        log("[" .. i .. "/" .. #files .. "] " .. filename)
        render_file(filename)
    end
    
    -- Calculate elapsed time
    local elapsed = reaper.time_precise() - start_time
    local minutes = math.floor(elapsed / 60)
    local seconds = math.floor(elapsed % 60)
    
    -- Summary
    log("")
    log("=" .. string.rep("=", 58))
    log("✓ COMPLETE!")
    log("=" .. string.rep("=", 58))
    log("Files rendered:   " .. #files)
    log("Time elapsed:     " .. minutes .. "m " .. seconds .. "s")
    log("Output location:  " .. OUTPUT_FOLDER)
    log("")
    
    -- Success dialog
    reaper.MB(
        "Batch render complete!\n\n" ..
        "Rendered: " .. #files .. " files\n" ..
        "Time: " .. minutes .. "m " .. seconds .. "s\n\n" ..
        "Output: " .. OUTPUT_FOLDER,
        "Success",
        0
    )
end

-- ============================================================
-- EXECUTION
-- ============================================================

reaper.ClearConsole()
main()
reaper.UpdateArrange()