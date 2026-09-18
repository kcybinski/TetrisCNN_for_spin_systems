import matplotlib.colors as mcolors
import colorsys

#colors = plt.get_cmap("tab10").colors
COLORS = {
    "green":       "#7FDF57",
    "green_dark":  "#4DAA2A",
    "green_light": "#B7F199",

    "blue":        "#3E52E1",
    "blue_dark":   "#2638B8",
    "blue_light":  "#7B8CFF",

    "red":         "#FF4D4D",
    "red_dark":    "#CC2F2F",
    "red_light":   "#FF8A8A",

    "purple":      "#9B59B6",
    "orange":      "#F5A623",
    "cyan":        "#2CB1BC",
    "gray":        "#7A7A7A",

    "magenta":     "#FC80FF",
    "magenta_dark": "#8B388D",
    "magenta_light":"#CDACCE",

    "blackish":     "#333333",
    "reddish":      "#B22222",

    "blue haze": "#ACC4C6",
    "hydro": "#3f6f75",
    "lagoon": "#018d8c",
    "green glow": "#a7cf61",
    "botanical garden": "#023333"
}
colors = [
    COLORS['blue'],
    COLORS['green_dark'],COLORS['green'],COLORS['red_dark'],COLORS['red'],
    COLORS['purple'], COLORS['orange'], COLORS['cyan'], COLORS['red_light'],
    COLORS['gray']
]

def adjust_lightness(color, factor=1.0):
    """
    factor > 1  -> lighter
    factor < 1  -> darker
    """
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0, min(1, l * factor))
    return colorsys.hls_to_rgb(h, l, s)

from matplotlib.colors import LinearSegmentedColormap
blue_cmap = LinearSegmentedColormap.from_list(
    "custom_blue",
    [
        (0.0, "#FFFFFF"),
        (0.4, "#7B8CFF"),
        (1.0, "#3E52E1"),
    ]
)