import argparse
import csv
import math
from pathlib import Path

import numpy as np
import tkinter as tk
import tkinter.font as tkfont
from PIL import Image, ImageTk

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / 'ground_truth'
LABEL_FRAME_US = 10000

import event_loader

ON_COLOR = np.array([70, 220, 70], dtype=np.float32)
OFF_COLOR = np.array([220, 70, 70], dtype=np.float32)
BOX_COLOR = '#00e5ff'
UNCERTAIN_COLOR = 'orange'
DRAG_COLOR = 'yellow'
HINT_COLOR = '#ff4fd8'
UNCERTAIN_DASH = (6, 3)
HINT_DASH = (5, 3)

MUTED_FG = '#6e6e73'
HELP_LINE = '#d2d2d7'
KEYCAP_BG = '#f5f5f7'
KEYCAP_FG = '#1d1d1f'
KEYCAP_BORDER = '#c7c7cc'
OVERLAY_BG = '#1c1c1e'
OVERLAY_BORDER = '#3a3a3c'
OVERLAY_FG = '#f5f5f7'
OVERLAY_MUTED = '#8e8e93'

HANDLE_PX = 4    # visual size (half-side) of the corner-drag handles, in canvas px
HIT_PX = 8       # how close the mouse must be to a corner/edge/handle to grab it

# hit_test() 'extra' tuples -> resize cursor to show for that corner
CORNER_CURSORS = {
    ('x1', 'y1'): 'top_left_corner',
    ('x2', 'y1'): 'top_right_corner',
    ('x1', 'y2'): 'bottom_left_corner',
    ('x2', 'y2'): 'bottom_right_corner',
}

KEY_LEGEND = (
    'drag empty area = new box    drag a corner/edge = resize    drag inside = move\n'
    'right-click a box (or hover + delete/backspace) = delete it    z = undo last box    c = clear boxes here\n'
    'u = toggle certain/uncertain on the hovered box (or the last box); uncertain boxes are orange dashed\n'
    'click a pink dashed box (--hint) = accept it\n'
    'enter/space/-> = confirm + next    <- = back    [ / ] = jump 20 frames    home/end = first/last frame\n'
    ', / . = previous / next frame with labels    g = go to a time\n'
    's = save now    q / esc = save and quit    ? = show all controls'
)

# Tuples are keys, strings are mouse gestures. Keep in sync with KEY_LEGEND.
HELP_SECTIONS = (
    ('Mouse', (
        ('Drag on empty area', 'Draw a new box'),
        ('Drag a corner or edge', 'Resize the box'),
        ('Drag inside a box', 'Move the box'),
        ('Right-click a box', 'Delete it'),
        ('Click a pink dashed box', 'Accept the hint (--hint)'),
    )),
    ('Boxes', (
        (('delete', 'backspace'), 'Delete the box under the mouse'),
        (('u',), 'Toggle certain / uncertain (hovered or last box)'),
        (('z',), 'Undo the last box on this frame'),
        (('c',), 'Clear all boxes on this frame'),
    )),
    ('Navigation', (
        (('enter', 'space', '→'), 'Confirm and go to the next frame'),
        (('←',), 'Go back one frame'),
        (('[', ']'), 'Jump 20 frames back / forward'),
        (('home', 'end'), 'First / last frame'),
        ((',', '.'), 'Previous / next frame with labels'),
        (('g',), 'Go to a time: 1500, 1.5s or 0:01.5'),
    )),
    ('File', (
        (('s',), 'Save now'),
        (('q', 'esc'), 'Save and quit'),
        (('?',), 'Show this window'),
    )),
)


def derived_font(name, **options):
    font = tkfont.nametofont(name).copy()
    font.configure(**options)
    return font


def keycap(parent, text, font):
    border = tk.Frame(parent, bg=KEYCAP_BORDER)
    tk.Label(border, text=text, font=font, bg=KEYCAP_BG, fg=KEYCAP_FG, padx=6).pack(padx=1, pady=1)
    return border


def swatch(parent, color, filled=False, dash=()):
    """A small black tile drawn the way the canvas draws an event or a box."""
    tile = tk.Canvas(parent, width=26, height=18, bg='black', highlightthickness=0)
    if filled:
        tile.create_rectangle(9, 5, 17, 13, fill=color, outline='')
    else:
        tile.create_rectangle(4, 3, 22, 15, outline=color, width=2, dash=dash)
    return tile


def count(n, singular, plural):
    return f'{n} {singular if n == 1 else plural}'


def rounded_rect(x0, y0, x1, y1, r):
    """Points for a rounded rectangle, to draw as a polygon with smooth=True.
    Tk keeps the outline straight between duplicated points and curves it
    around the single corner points."""
    edges = ((x0 + r, y0), (x1 - r, y0), (x1, y0 + r), (x1, y1 - r),
             (x1 - r, y1), (x0 + r, y1), (x0, y1 - r), (x0, y0 + r))
    corners = ((x1, y0), (x1, y1), (x0, y1), (x0, y0))
    points = []
    for i, corner in enumerate(corners):
        start, end = edges[2 * i], edges[2 * i + 1]
        points += [*start, *start, *end, *end, *corner]
    return points


def parse_time_ms(text):
    """'1500', '1500ms', '1.5s' or 'm:ss.fff' -> milliseconds; None if it can't be read."""
    text = text.strip().lower().replace(' ', '')
    try:
        if ':' in text:
            minutes, seconds = text.split(':')
            ms = (int(minutes or 0) * 60 + float(seconds)) * 1000
        elif text.endswith('ms'):
            ms = float(text[:-2])
        elif text.endswith('s'):
            ms = float(text[:-1]) * 1000
        else:
            ms = float(text)
    except ValueError:
        return None
    return ms if math.isfinite(ms) else None


def rgb_hex(color):
    return '#%02x%02x%02x' % tuple(int(c) for c in color)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Step through an event-camera recording at a fixed interval and '
                     'manually draw bounding boxes, to build ground-truth labels for '
                     'automating evaluation of a detector/tracker.',
        epilog=KEY_LEGEND,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source', help='event file to label (.aedat4, .es, .raw, .dat, .csv, ...)')
    parser.add_argument('-o', '--output', default=None,
                         help='output CSV path (default: ground_truth/<source stem>.csv next to this script)')
    parser.add_argument('--seed', default=None,
                         help='predictions or label CSV to pre-populate boxes '
                              'from, so you correct/confirm tracker output instead of drawing from '
                              'scratch; frames already saved in the output file take precedence')
    parser.add_argument('--hint', default=None,
                         help='CSV of candidate boxes (same formats as --seed), drawn as pink dashed '
                              'boxes; they are not saved unless you click one to accept it')
    parser.add_argument('--class-name', default='insect',
                         help='value written to the class column of the output CSV (default: insect)')
    parser.add_argument('--step-ms', type=float, default=10.0,
                         help='how far playback advances per stop, in ms; must be a multiple of '
                              'the 10 ms label frame (default: 10)')
    parser.add_argument('--context-ms', type=float, default=25.0,
                         help='how much event history is shown at each stop, in ms (default: 25)')
    parser.add_argument('--start-ms', type=float, default=None,
                         help='timestamp to start labeling at, in ms relative to the start of '
                              'the recording (default: resume after the last saved label, or 0)')
    parser.add_argument('--mask-hot-pixels', metavar='FRAC', nargs='?', const=0.5, default=None, type=float,
                        help='hide pixels active in >= FRAC of 1 s bins (default 0.5)')
    parser.add_argument('--width', type=int, default=640, help='sensor width fallback (default: 640)')
    parser.add_argument('--height', type=int, default=480, help='sensor height fallback (default: 480)')
    parser.add_argument('--scale', type=float, default=None,
                         help='display scale factor (default: auto, fits a ~900px-tall window)')
    args = parser.parse_args()
    if args.step_ms <= 0:
        raise SystemExit('error: --step-ms must be > 0')
    if round(args.step_ms * 1000) % LABEL_FRAME_US:
        raise SystemExit('error: --step-ms must be a multiple of 10 (labels are saved as 10 ms frames)')
    if args.context_ms <= 0:
        raise SystemExit('error: --context-ms must be > 0')
    return args


def load_events(path, mask_hot_pixels=None):
    """Reads any supported event file/CSV into time-sorted, zero-based (t[0]=0)
    t (int64 us), x, y (uint16), p (bool) arrays, matching the
    event_loader.read convention."""
    path = Path(path)
    if path.suffix.lower() == '.csv':
        # Columns: x, y, polarity, timestamp_us.
        x, y, p, t = np.loadtxt(path, delimiter=',', dtype=np.int64, unpack=True)
    else:
        data = event_loader.read(str(path))
        if mask_hot_pixels is not None:
            data = event_loader.drop_hot_pixels(data, mask_hot_pixels)
        t, x, y, p = data.t.astype(np.int64), data.x, data.y, data.p
    if len(t) == 0:
        return t.astype(np.int64), x.astype(np.uint16), y.astype(np.uint16), p.astype(np.bool_)
    order = np.argsort(t, kind='stable')
    t, x, y, p = t[order], x[order], y[order], p[order]
    t = (t - t[0]).astype(np.int64)
    return t, x.astype(np.uint16), y.astype(np.uint16), p.astype(np.bool_)


def render_window(t, x, y, p, win_start, win_end, width, height, fade):
    """Composites events in [win_start, win_end) into an RGB frame: green for ON,
    red for OFF. When fade is set, older events within the window are dimmer;
    at equal pixels the most recent event wins (arrays are time-sorted)."""
    lo, hi = np.searchsorted(t, (win_start, win_end))
    canvas = np.zeros((height, width, 3), dtype=np.float32)
    if hi > lo:
        xs, ys, ps = x[lo:hi], y[lo:hi], p[lo:hi]
        colors = np.where(ps[:, None], ON_COLOR, OFF_COLOR)
        if fade:
            age = (win_end - t[lo:hi]).astype(np.float32) / (win_end - win_start)
            colors = colors * (1.0 - 0.85 * age)[:, None]
        canvas[ys, xs] = colors
    return np.clip(canvas, 0, 255).astype(np.uint8)


class LabelingApp:
    def __init__(self, args, t, x, y, p, width, height, out_path):
        self.t, self.x, self.y, self.p = t, x, y, p
        self.width, self.height = width, height
        self.out_path = out_path
        self.class_name = args.class_name

        self.step_us = int(round(args.step_ms * 1000))
        self.context_us = int(round(args.context_ms * 1000))
        self.fade = self.context_us > self.step_us

        total_us = int(t[-1]) if len(t) else 0
        self.n_frames = max(1, math.ceil(total_us / self.step_us))

        self.labels = {}  # frame_index -> list of (x1, y1, x2, y2, confidence)
        if args.seed:
            self.labels = self.load_seed_predictions(Path(args.seed))
        last_saved_frame = self.load_existing_labels()
        self.hints = self.load_seed_predictions(Path(args.hint)) if args.hint else {}

        if args.start_ms is not None:
            start_frame = int(round(args.start_ms * 1000)) // self.step_us
        elif last_saved_frame is not None:
            start_frame = last_saved_frame + 1
        else:
            start_frame = 0
        self.frame_index = max(0, min(start_frame, self.n_frames - 1))

        self.scale = args.scale if args.scale else min(3.0, max(1.0, 900.0 / height))

        self.drag_start = None
        self.drag_mode = None      # 'new', 'move', 'corner', or 'edge'
        self.edit_index = None     # index into self.labels[frame_index] being edited
        self.edit_frame = None     # frame the edited box belongs to
        self.edit_fields = None    # which of x1/y1/x2/y2 the current handle controls
        self.edit_orig = None      # box (x1, y1, x2, y2, confidence) as it was when the drag started
        self.temp_rect = None
        self.hover_hit = None      # hit_test() result under the mouse, kept fresh by on_motion
        self.box_items = []
        self.photo = None
        self.image_id = None
        self.help_win = None

        self.build_ui(args.source)
        self.draw_frame()

    def build_ui(self, source_name):
        self.root = tk.Tk()
        self.root.title(f'Label events: {source_name}  →  {self.out_path.name}')
        self.allow_uncertain = tk.BooleanVar(self.root, value=True)

        self.canvas_w = int(round(self.width * self.scale))
        self.canvas_h = int(round(self.height * self.scale))
        self.canvas = tk.Canvas(self.root, width=self.canvas_w, height=self.canvas_h,
                                 bg='black', highlightthickness=0, cursor='tcross')
        self.canvas.pack()

        bar = tk.Frame(self.root, padx=8, pady=6)
        bar.pack(fill='x')
        for text, command in (('◀', self.on_prev), ('▶', self.on_confirm)):
            tk.Button(bar, text=text, width=2, takefocus=0, command=command).pack(side='left')
        # Packed before the status labels, so a narrow window clips the labels, not the buttons.
        for text, command in (('?', self.show_help),
                              ('clear', self.on_clear),
                              ('labeled ▶', lambda: self.jump_labeled(1)),
                              ('◀ labeled', lambda: self.jump_labeled(-1))):
            tk.Button(bar, text=text, takefocus=0, command=command).pack(side='right')
        self.time_entry = tk.Entry(bar, width=9)
        self.time_entry.pack(side='right', padx=(4, 12))
        tk.Label(bar, text='go to', fg=MUTED_FG).pack(side='right')
        # Leaving out the main window's bindtag, so typing here doesn't fire shortcuts (s, ., backspace, ...).
        self.time_entry.bindtags((str(self.time_entry), 'Entry', 'all'))
        for key in ('<Return>', '<KP_Enter>'):
            self.time_entry.bind(key, self.on_time_entry)
        self.time_entry.bind('<Escape>', self.leave_time_entry)

        counter_width = len(f'frame {self.n_frames} / {self.n_frames}')
        self.frame_label = tk.Label(bar, font=derived_font('TkDefaultFont', weight='bold'),
                                    width=counter_width, anchor='w')
        self.frame_label.pack(side='left', padx=(10, 0))
        self.here_label = tk.Label(bar)
        self.here_label.pack(side='left')
        tk.Label(bar, text='·', fg=MUTED_FG).pack(side='left', padx=10)
        self.total_label = tk.Label(bar, fg=MUTED_FG)
        self.total_label.pack(side='left')

        self.timestamp_bg = self.canvas.create_polygon(
            *rounded_rect(0, 0, 0, 0, 0), smooth=True, fill=OVERLAY_BG, outline=OVERLAY_BORDER)
        self.timestamp_text = self.canvas.create_text(
            0, 0, anchor='w', fill=OVERLAY_FG, font=derived_font('TkFixedFont', size=12, weight='bold'))
        self.timestamp_clock = self.canvas.create_text(
            0, 0, anchor='w', fill=OVERLAY_MUTED, font=derived_font('TkFixedFont', size=10))

        self.canvas.bind('<ButtonPress-1>', self.on_press)
        self.canvas.bind('<B1-Motion>', self.on_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_release)
        self.canvas.bind('<Motion>', self.on_motion)
        # Button-2/3 covers right-click on X11/Windows vs. macOS Tk; Control-Button-1
        # covers a trackpad secondary-click that arrives as a modified left click.
        for button in ('<Button-2>', '<Button-3>', '<Control-Button-1>'):
            self.canvas.bind(button, self.on_delete_click)

        for key in ('<Return>', '<space>', '<Right>'):
            self.root.bind(key, self.on_confirm)
        self.root.bind('<Left>', self.on_prev)
        self.root.bind('z', self.on_undo)
        self.root.bind('c', self.on_clear)
        self.root.bind('u', self.on_toggle_confidence)
        self.root.bind('<Delete>', self.on_delete_key)
        self.root.bind('<BackSpace>', self.on_delete_key)
        self.root.bind('[', lambda e: self.move_to(self.frame_index - 20))
        self.root.bind(']', lambda e: self.move_to(self.frame_index + 20))
        self.root.bind('<Home>', lambda e: self.move_to(0))
        self.root.bind('<End>', lambda e: self.move_to(self.n_frames - 1))
        self.root.bind(',', lambda e: self.jump_labeled(-1))
        self.root.bind('.', lambda e: self.jump_labeled(1))
        self.root.bind('s', lambda e: self.save())
        self.root.bind('<question>', self.show_help)
        self.root.bind('g', self.focus_time_entry)
        self.root.bind('q', self.on_quit)
        self.root.bind('<Escape>', self.on_quit)
        self.root.protocol('WM_DELETE_WINDOW', self.on_quit)

    def show_help(self, event=None):
        if self.help_win is not None and self.help_win.winfo_exists():
            self.help_win.lift()
            self.help_win.focus_set()
            return
        # A separate toplevel, so the main window's keys (q, esc, ...) don't fire while it has focus.
        win = tk.Toplevel(self.root)
        win.title('Controls')
        win.transient(self.root)
        win.resizable(False, False)

        text_font = derived_font('TkDefaultFont')
        bold_font = derived_font('TkDefaultFont', weight='bold')
        header_font = derived_font('TkDefaultFont', size=11, weight='bold')
        small_font = derived_font('TkDefaultFont', size=11)
        key_font = derived_font('TkFixedFont', size=12)

        body = tk.Frame(win, padx=24, pady=20)
        body.pack(fill='both', expand=True)

        def header(parent, text, **grid):
            label = tk.Label(parent, text=text.upper(), font=header_font, fg=MUTED_FG)
            if grid:
                label.grid(sticky='w', **grid)
            else:
                label.pack(anchor='w', pady=(0, 8))

        def separator():
            tk.Frame(body, height=1, bg=HELP_LINE).pack(fill='x', pady=16)

        columns = tk.Frame(body)
        columns.pack(fill='x')
        half = (len(HELP_SECTIONS) + 1) // 2
        for col, sections in enumerate((HELP_SECTIONS[:half], HELP_SECTIONS[half:])):
            grid = tk.Frame(columns)
            grid.grid(row=0, column=col, sticky='nw', padx=(0, 40) if col == 0 else 0)
            r = 0
            for section, entries in sections:
                header(grid, section, row=r, column=0, columnspan=2, pady=(0 if r == 0 else 18, 6))
                r += 1
                for keys, action in entries:
                    if isinstance(keys, str):
                        what = tk.Label(grid, text=keys, font=bold_font)
                    else:
                        what = tk.Frame(grid)
                        for key in keys:
                            keycap(what, key, key_font).pack(side='left', padx=(0, 4))
                    what.grid(row=r, column=0, sticky='w', pady=3)
                    tk.Label(grid, text=action, font=text_font).grid(
                        row=r, column=1, sticky='w', padx=(14, 0))
                    r += 1

        separator()
        header(body, 'Colors')
        legend = tk.Frame(body)
        legend.pack(anchor='w')
        for color, filled, dash, meaning in (
                (rgb_hex(ON_COLOR), True, (), 'ON event'),
                (rgb_hex(OFF_COLOR), True, (), 'OFF event'),
                (BOX_COLOR, False, (), 'Certain box'),
                (UNCERTAIN_COLOR, False, UNCERTAIN_DASH, 'Uncertain box'),
                (HINT_COLOR, False, HINT_DASH, 'Hint (not saved until clicked)')):
            swatch(legend, color, filled, dash).pack(side='left')
            tk.Label(legend, text=meaning, font=text_font).pack(side='left', padx=(6, 20))

        separator()
        header(body, 'Settings')
        tk.Checkbutton(body, text='Allow uncertain boxes', font=text_font, takefocus=0,
                       variable=self.allow_uncertain, command=self.on_uncertain_setting).pack(anchor='w')
        tk.Label(body, text='When off, u does nothing and every box is saved as certain.',
                 font=small_font, fg=MUTED_FG).pack(anchor='w', padx=(24, 0))

        separator()
        try:
            shown_path = self.out_path.resolve().relative_to(Path.cwd())
        except ValueError:
            shown_path = self.out_path
        footer = tk.Frame(body)
        footer.pack(fill='x')
        tk.Label(footer, justify='left', font=small_font, fg=MUTED_FG,
                 text=f'Labels are saved to {shown_path} on every frame change and on quit.\n'
                      'Run with -h for the command-line options.').pack(side='left')
        tk.Button(footer, text='Close', takefocus=0, command=self.close_help).pack(side='right')

        for key in ('<Escape>', '<question>'):
            win.bind(key, self.close_help)
        win.protocol('WM_DELETE_WINDOW', self.close_help)
        self.help_win = win
        win.focus_set()

    def close_help(self, event=None):
        if self.help_win is not None:
            self.help_win.destroy()
            self.help_win = None
        self.root.focus_set()

    def stop_time(self, frame_index):
        return (frame_index + 1) * self.step_us

    def current_window(self):
        win_end = self.stop_time(self.frame_index)
        return win_end - self.context_us, win_end

    def draw_frame(self):
        win_start, win_end = self.current_window()
        frame = render_window(self.t, self.x, self.y, self.p, win_start, win_end,
                               self.width, self.height, self.fade)
        img = Image.fromarray(frame, mode='RGB').resize((self.canvas_w, self.canvas_h), Image.NEAREST)
        self.photo = ImageTk.PhotoImage(img)
        if self.image_id is None:
            self.image_id = self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        else:
            self.canvas.itemconfig(self.image_id, image=self.photo)
        self.canvas.tag_lower(self.image_id)
        self.redraw_boxes()
        self.update_status()

    def redraw_boxes(self):
        for item in self.box_items:
            self.canvas.delete(item)
        self.box_items = []
        for x1, y1, x2, y2, _ in self.visible_hints():
            self.box_items.append(self.canvas.create_rectangle(
                x1 * self.scale, y1 * self.scale, x2 * self.scale, y2 * self.scale,
                outline=HINT_COLOR, width=2, dash=HINT_DASH))
        for x1, y1, x2, y2, conf in self.labels.get(self.frame_index, []):
            bx1, by1, bx2, by2 = x1 * self.scale, y1 * self.scale, x2 * self.scale, y2 * self.scale
            certain = self.effective_conf(conf) == 'certain'
            color = BOX_COLOR if certain else UNCERTAIN_COLOR
            self.box_items.append(self.canvas.create_rectangle(
                bx1, by1, bx2, by2, outline=color, width=2, dash=() if certain else UNCERTAIN_DASH))
            for hx, hy in ((bx1, by1), (bx2, by1), (bx1, by2), (bx2, by2)):
                self.box_items.append(self.canvas.create_rectangle(
                    hx - HANDLE_PX, hy - HANDLE_PX, hx + HANDLE_PX, hy + HANDLE_PX,
                    outline=color, fill='black'))

    def visible_hints(self):
        """Hints on this frame that no label box covers yet."""
        labels = self.labels.get(self.frame_index, [])
        out = []
        for h in self.hints.get(self.frame_index, []):
            cx, cy = (h[0] + h[2]) / 2, (h[1] + h[3]) / 2
            if not any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in labels):
                out.append(h)
        return out

    def hint_at(self, cx, cy):
        """The smallest visible hint under canvas point (cx, cy), or None."""
        x, y = cx / self.scale, cy / self.scale
        under = [h for h in self.visible_hints() if h[0] <= x <= h[2] and h[1] <= y <= h[3]]
        return min(under, key=lambda h: (h[2] - h[0]) * (h[3] - h[1]), default=None)

    def update_status(self):
        stop_us = self.stop_time(self.frame_index)
        mm, ss = divmod(stop_us / 1e6, 60)
        here = self.labels.get(self.frame_index, [])
        uncertain_here = sum(1 for b in here if self.effective_conf(b[4]) != 'certain')
        labeled_frames = sum(1 for v in self.labels.values() if v)
        total_boxes = sum(len(v) for v in self.labels.values())
        here_text = f'{count(len(here), "box", "boxes")} here'
        if uncertain_here:
            here_text += f' ({uncertain_here} uncertain)'
        self.frame_label.config(text=f'frame {self.frame_index + 1} / {self.n_frames}')
        self.here_label.config(text=here_text)
        self.total_label.config(
            text=f'{count(total_boxes, "box", "boxes")} in {count(labeled_frames, "frame", "frames")}')

        self.canvas.itemconfig(self.timestamp_text, text=f'{stop_us / 1000:.1f} ms')
        self.canvas.itemconfig(self.timestamp_clock, text=f'{int(mm):02d}:{ss:06.3f}')
        self.layout_timestamp()

    def layout_timestamp(self):
        margin, pad_x, pad_y, gap = 8, 7, 3, 7
        c = self.canvas
        c.coords(self.timestamp_text, margin + pad_x, 0)
        _, top, right, _ = c.bbox(self.timestamp_text)
        mid = margin + pad_y - top
        c.coords(self.timestamp_text, margin + pad_x, mid)
        c.coords(self.timestamp_clock, right + gap, mid)
        _, _, x1, y1 = c.bbox(self.timestamp_text, self.timestamp_clock)
        c.coords(self.timestamp_bg, *rounded_rect(margin, margin, x1 + pad_x, y1 + pad_y, 6))
        for item in (self.timestamp_bg, self.timestamp_text, self.timestamp_clock):
            c.tag_raise(item)

    def hit_test(self, cx, cy):
        """Finds what canvas point (cx, cy) would grab on the current frame's
        boxes: a corner or edge handle to resize, the box interior to move, or
        nothing (empty canvas, so a new box should be started). Boxes are
        checked topmost (last-drawn) first."""
        boxes = self.labels.get(self.frame_index, [])
        for i in range(len(boxes) - 1, -1, -1):
            x1, y1, x2, y2, _ = boxes[i]
            bx1, by1, bx2, by2 = x1 * self.scale, y1 * self.scale, x2 * self.scale, y2 * self.scale
            if not (bx1 - HIT_PX <= cx <= bx2 + HIT_PX and by1 - HIT_PX <= cy <= by2 + HIT_PX):
                continue
            near_left, near_right = abs(cx - bx1) <= HIT_PX, abs(cx - bx2) <= HIT_PX
            near_top, near_bottom = abs(cy - by1) <= HIT_PX, abs(cy - by2) <= HIT_PX
            if near_left and near_top:
                return i, 'corner', ('x1', 'y1')
            if near_right and near_top:
                return i, 'corner', ('x2', 'y1')
            if near_left and near_bottom:
                return i, 'corner', ('x1', 'y2')
            if near_right and near_bottom:
                return i, 'corner', ('x2', 'y2')
            if near_left:
                return i, 'edge', ('x1',)
            if near_right:
                return i, 'edge', ('x2',)
            if near_top:
                return i, 'edge', ('y1',)
            if near_bottom:
                return i, 'edge', ('y2',)
            if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                return i, 'move', None
        return None

    def focus_time_entry(self, event=None):
        self.time_entry.focus_set()
        self.time_entry.select_range(0, 'end')

    def on_time_entry(self, event=None):
        ms = parse_time_ms(self.time_entry.get())
        if ms is None:
            self.root.bell()
            self.time_entry.select_range(0, 'end')
            return
        self.leave_time_entry()
        # The first stop at or after that time, so the overlay shows what was typed.
        self.move_to(math.ceil(round(ms * 1000) / self.step_us) - 1)

    def leave_time_entry(self, event=None):
        self.time_entry.delete(0, 'end')
        self.root.focus_set()

    def on_press(self, event):
        # The canvas never takes focus, so clicking it would otherwise leave keys going to the time field.
        self.root.focus_set()
        self.drag_start = (event.x, event.y)
        hit = self.hit_test(event.x, event.y)
        if hit is None:
            self.drag_mode = 'new'
            self.temp_rect = self.canvas.create_rectangle(
                event.x, event.y, event.x, event.y, outline=DRAG_COLOR, width=2, dash=(4, 2))
            return
        self.edit_index, self.drag_mode, self.edit_fields = hit
        self.edit_frame = self.frame_index
        self.edit_orig = self.labels[self.frame_index][self.edit_index]

    def on_drag(self, event):
        if self.drag_start is None:
            return
        if self.drag_mode == 'new':
            x0, y0 = self.drag_start
            self.canvas.coords(self.temp_rect, x0, y0, event.x, event.y)
            return

        # The box was deleted, or the frame changed, while the button was down.
        boxes = self.labels.get(self.frame_index)
        if self.frame_index != self.edit_frame or not boxes or self.edit_index >= len(boxes):
            self.drag_start = None
            self.drag_mode = None
            return

        dx =(event.x - self.drag_start[0]) / self.scale
        dy = (event.y - self.drag_start[1]) / self.scale
        x1, y1, x2, y2, conf = self.edit_orig

        if self.drag_mode == 'move':
            w, h = x2 - x1, y2 - y1
            x1 = np.clip(x1 + dx, 0, self.width - w)
            y1 = np.clip(y1 + dy, 0, self.height - h)
            x2, y2 = x1 + w, y1 + h
        else:
            if 'x1' in self.edit_fields:
                x1 = np.clip(x1 + dx, 0, x2 - 1)
            if 'x2' in self.edit_fields:
                x2 = np.clip(x2 + dx, x1 + 1, self.width)
            if 'y1' in self.edit_fields:
                y1 = np.clip(y1 + dy, 0, y2 - 1)
            if 'y2' in self.edit_fields:
                y2 = np.clip(y2 + dy, y1 + 1, self.height)

        box = tuple(int(round(v)) for v in (x1, y1, x2, y2))
        self.labels[self.frame_index][self.edit_index] = box + (conf,)
        self.redraw_boxes()

    def on_release(self, event):
        if self.drag_start is None:
            return
        x0, y0 = self.drag_start
        self.drag_start = None

        if self.drag_mode != 'new':
            self.drag_mode = None
            self.edit_index = None
            self.edit_fields = None
            self.edit_orig = None
            self.update_status()
            return

        self.canvas.delete(self.temp_rect)
        self.temp_rect = None
        self.drag_mode = None

        cx0, cx1 = sorted((x0, event.x))
        cy0, cy1 = sorted((y0, event.y))
        if cx1 - cx0 < 4 or cy1 - cy0 < 4:
            # A click, not a drag: accept the hint under it, if any.
            hint = self.hint_at(x0, y0)
            if hint is not None:
                self.labels.setdefault(self.frame_index, []).append(tuple(hint))
                self.redraw_boxes()
                self.update_status()
            return

        x1 = int(np.clip(round(cx0 / self.scale), 0, self.width - 1))
        y1 = int(np.clip(round(cy0 / self.scale), 0, self.height - 1))
        x2 = int(np.clip(round(cx1 / self.scale), x1 + 1, self.width))
        y2 = int(np.clip(round(cy1 / self.scale), y1 + 1, self.height))
        self.labels.setdefault(self.frame_index, []).append((x1, y1, x2, y2, 'certain'))
        self.redraw_boxes()
        self.update_status()

    def on_motion(self, event):
        if self.drag_start is not None:
            return  # mid-drag: keep whatever cursor was set on press
        hit = self.hit_test(event.x, event.y)
        self.hover_hit = hit
        if hit is None:
            cursor = 'tcross'
        else:
            _, mode, extra = hit
            if mode == 'move':
                cursor = 'fleur'
            elif mode == 'corner':
                cursor = CORNER_CURSORS[extra]
            else:
                cursor = 'sb_h_double_arrow' if extra == ('x1',) or extra == ('x2',) else 'sb_v_double_arrow'
        self.canvas.config(cursor=cursor)

    def on_delete_click(self, event):
        self.delete_box(self.hit_test(event.x, event.y))

    def on_delete_key(self, event=None):
        self.delete_box(self.hover_hit)

    def delete_box(self, hit):
        if hit is None:
            return
        index = hit[0]
        boxes = self.labels.get(self.frame_index)
        if not boxes or not (0 <= index < len(boxes)):
            return
        boxes.pop(index)
        if not boxes:
            del self.labels[self.frame_index]
        self.hover_hit = None
        self.redraw_boxes()
        self.update_status()

    def effective_conf(self, conf):
        # Stored confidences are left alone, so turning the setting back on restores them.
        return conf if self.allow_uncertain.get() else 'certain'

    def on_uncertain_setting(self):
        self.redraw_boxes()
        self.update_status()

    def on_toggle_confidence(self, event=None):
        boxes = self.labels.get(self.frame_index)
        if not boxes or not self.allow_uncertain.get():
            return
        index = self.hover_hit[0] if self.hover_hit is not None else len(boxes) - 1
        if not (0 <= index < len(boxes)):
            return
        *coords, conf = boxes[index]
        boxes[index] = (*coords, 'uncertain' if conf == 'certain' else 'certain')
        self.redraw_boxes()
        self.update_status()

    def on_undo(self, event=None):
        boxes = self.labels.get(self.frame_index)
        if not boxes:
            return
        boxes.pop()
        if not boxes:
            del self.labels[self.frame_index]
        self.redraw_boxes()
        self.update_status()

    def on_clear(self, event=None):
        if self.labels.pop(self.frame_index, None) is not None:
            self.redraw_boxes()
            self.update_status()

    def on_confirm(self, event=None):
        self.move_to(self.frame_index + 1)

    def on_prev(self, event=None):
        self.move_to(self.frame_index - 1)

    def move_to(self, index):
        self.save()
        self.frame_index = max(0, min(index, self.n_frames - 1))
        self.draw_frame()

    def jump_labeled(self, direction):
        # Go to the closest frame with boxes before (-1) or after (+1) this one.
        frames = [i for i, boxes in self.labels.items()
                  if boxes and (i - self.frame_index) * direction > 0]
        if frames:
            self.move_to(min(frames) if direction > 0 else max(frames))

    def on_quit(self, event=None):
        self.save()
        labeled_frames = sum(1 for v in self.labels.values() if v)
        total_boxes = sum(len(v) for v in self.labels.values())
        print(f'Saved {total_boxes} boxes across {labeled_frames} frames -> {self.out_path}')
        self.root.destroy()

    def load_existing_labels(self):
        """Merges saved ground truth into self.labels, overwriting seeded
        predictions for covered frames. Returns the highest saved frame
        index, or None if there's no output file yet."""
        if not self.out_path.exists():
            return None
        with open(self.out_path, newline='') as f:
            rows = list(csv.reader(f))
        has_conf = bool(rows) and rows[0] == ['timestamp_us', 'x1', 'y1', 'x2', 'y2', 'confidence']
        if rows and (has_conf or rows[0] == ['timestamp_us', 'x1', 'y1', 'x2', 'y2']):
            rows = rows[1:]
        elif rows and rows[0] == ['timestamp', 'track_id', 'x1', 'y1', 'x2', 'y2']:
            # Output path points at a not-yet-reviewed predictions CSV (e.g. --seed
            # and --output are the same file): nothing saved yet, so nothing to resume.
            return None
        elif rows and 'frame_index' in rows[0]:
            # Label file, the format save() writes.
            saved = self.load_seed_predictions(self.out_path)
            self.labels.update(saved)
            return max(saved) if saved else None
        saved = {}
        for row in rows:
            if not row:
                continue
            ts, x1, y1, x2, y2 = (int(float(v)) for v in row[:5])
            conf = row[5] if has_conf and len(row) > 5 and row[5] else 'certain'
            frame_index = round(ts / self.step_us) - 1
            saved.setdefault(frame_index, []).append((x1, y1, x2, y2, conf))
        self.labels.update(saved)
        return max(saved) if saved else None

    def load_seed_predictions(self, path):
        """Loads seed boxes from a tracker predictions CSV
        (timestamp,track_id,x1,y1,x2,y2), a label CSV
        (frame_index,left,top,width,height; 10 ms frames) or an old-format one
        (timestamp_us,x1,y1,x2,y2) into
        frame_index -> list of boxes. A `confidence` column is kept if present,
        otherwise boxes are `certain`."""
        if not path.exists():
            return {}
        boxes_by_frame = {}
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            ts_key = 'timestamp' if 'timestamp' in fields else 'timestamp_us'
            for row in reader:
                if 'frame_index' in fields:
                    ts = (int(row['frame_index']) + 1) * LABEL_FRAME_US
                    x1, y1 = float(row['left']), float(row['top'])
                    coords = (x1, y1, x1 + float(row['width']), y1 + float(row['height']))
                else:
                    ts = int(float(row[ts_key]))
                    coords = tuple(float(row[k]) for k in ('x1', 'y1', 'x2', 'y2'))
                frame_index = round(ts / self.step_us) - 1
                conf = row.get('confidence') or 'certain'
                box = tuple(int(round(v)) for v in coords) + (conf,)
                boxes_by_frame.setdefault(frame_index, []).append(box)
        return boxes_by_frame

    def save(self):
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.out_path.with_suffix(self.out_path.suffix + '.tmp')
        with open(tmp_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame_index', 'is_keyframe', 'class', 'confidence',
                             'left', 'top', 'width', 'height', 'center_x', 'center_y'])
            for frame_index in sorted(self.labels):
                # Label frame n ends at (n + 1) * 10 ms.
                label_index = self.stop_time(frame_index) // LABEL_FRAME_US - 1
                for x1, y1, x2, y2, conf in self.labels[frame_index]:
                    writer.writerow([label_index, 'True', self.class_name, self.effective_conf(conf),
                                     float(x1), float(y1), float(x2 - x1), float(y2 - y1),
                                     (x1 + x2) / 2, (y1 + y2) / 2])
        tmp_path.replace(self.out_path)


def main():
    args = parse_args()
    source_path = Path(args.source)
    if not source_path.exists():
        raise SystemExit(f'error: source file not found: {source_path}')
    out_path = Path(args.output) if args.output else DEFAULT_OUTPUT_DIR / f'{source_path.stem}.csv'

    print(f'Loading {source_path} ...')
    t, x, y, p = load_events(source_path, args.mask_hot_pixels)
    width = max(args.width, int(x.max()) + 1) if len(x) else args.width
    height = max(args.height, int(y.max()) + 1) if len(y) else args.height
    if (width, height) != (args.width, args.height):
        print(f'note: canvas expanded to {width}x{height} to fit all events')
    duration_s = (int(t[-1]) if len(t) else 0) / 1e6
    print(f'{len(t)} events, {duration_s:.2f}s, {width}x{height} -> labels: {out_path}')
    print(KEY_LEGEND)

    app = LabelingApp(args, t, x, y, p, width, height, out_path)
    app.root.mainloop()


if __name__ == '__main__':
    main()
