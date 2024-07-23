# Vulkan Offline Rendering

![Offline Rendering Output](docs/offline.jpg)

## Overview

This sample demonstrates creating a Vulkan context, rendering, and saving results to disk without a window interface.

## Key Features

- Windowless Vulkan context creation
- Offline rendering pipeline
- Image output to disk

## Usage

```bash
offline.exe [OPTIONS]
```

### Options

| Flag | Long Form | Description |
|------|-----------|-------------|
| -t   | --time    | Animation time |
| -w   | --width   | Render width |
| -h   | --height  | Render height |
| -o   | --output  | Output filename (must be .jpg) |

## Technical Considerations

- Implements headless Vulkan instance
- Manages render targets and framebuffers without display
- Handles image data transfer from GPU to system memory
- Implements image encoding and file I/O for output
