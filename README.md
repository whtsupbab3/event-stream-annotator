# event-stream-annotator

A small tool for drawing bounding boxes on event-camera recordings, so you can
build ground-truth labels by hand. It moves through the recording in 10 ms steps
and shows the latest events at each step: green for ON, red for OFF.

## Setup

    pip install -r requirements.txt

You also need Tkinter, which comes with most Python installs.

**On macOS**, avoid the built-in `/usr/bin/python3`. Its Tk is too old, and the
window stays blank. Homebrew's Python works fine:

    /opt/homebrew/bin/python3.12 -m venv .venv
    .venv/bin/pip install -r requirements.txt

## Usage

    python label_events.py recording.aedat4
    python label_events.py recording.aedat4 --seed predictions.csv   # start from tracker output and fix it
    python label_events.py recording.aedat4 --hint candidates.csv    # show suggestions you can click to accept

On macOS, use `.venv/bin/python` instead of `python`.

It reads `.aedat4`, `.es`, `.raw` and `.dat` files (through faery), `.h5` files
with a `CD/events` dataset, and CSV files with `x,y,polarity,timestamp_us` columns.

Labels are saved to `ground_truth/<recording name>.csv` (change this with `-o`)
whenever you move to another frame and when you quit. When you open the same
recording again, you pick up where you left off.

Press `?` in the window to see all the controls. `-h` lists the command-line options.

## Output

One row per box. Frames are 10 ms long, and frame `n` ends at `(n + 1) × 10 ms`.

    frame_index,is_keyframe,class,confidence,left,top,width,height,center_x,center_y

`class` is `insect` by default. Set it with `--class-name`, e.g. `--class-name drone`.

`confidence` is either `certain` or `uncertain`. Press `u` over a box to switch
between them. If you don't need this, untick **Allow uncertain boxes** in the
Controls window, and every box is saved as `certain`.
