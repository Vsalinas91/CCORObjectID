from PIL import Image
from PIL import GifImagePlugin
import contextlib
import glob

GifImagePlugin.LOADING_STRATEGY = GifImagePlugin.LoadingStrategy.RGB_AFTER_DIFFERENT_PALETTE_ONLY


def convertToP(im):
    """
    Ensure quality of gif is same as images.
    src: https://github.com/python-pillow/Pillow/issues/6832
    """
    if im.getcolors() is not None:
        # There are 256 colors or less in this image
        p = Image.new("P", im.size)
        transparent_pixels = []
        for x in range(im.width):
            for y in range(im.height):
                pixel = im.getpixel((x, y))
                if pixel[3] == 0:
                    transparent_pixels.append((x, y))
                else:
                    color = p.palette.getcolor(pixel[:3])
                    p.putpixel((x, y), color)
        if transparent_pixels and len(p.palette.colors) < 256:
            color = (0, 0, 0)
            while color in p.palette.colors:
                if color[0] < 255:
                    color = (color[0] + 1, color[1], color[2])
                else:
                    color = (color[0], color[1] + 1, color[2])
            transparency = p.palette.getcolor(color)
            p.info["transparency"] = transparency
            for x, y in transparent_pixels:
                p.putpixel((x, y), transparency)
        return p
    return im.convert("P", palette=Image.Palette.ADAPTIVE)


def save_animation(path):
    fp_in = f"{path}/*.png"
    fp_out = f"{path}/ccor1.gif"

    # use exit stack to automatically close opened images
    with contextlib.ExitStack() as stack:

        # lazily load images
        imgs = (stack.enter_context(convertToP(Image.open(f).convert("RGB"))) for f in sorted(glob.glob(fp_in)))

        # extract  first image from iterator
        img = next(imgs)

        # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
        img.save(
            fp=fp_out,
            format="GIF",
            append_images=imgs,
            save_all=True,
            duration=60,
            loop=0,
            quality=100,
            optimize=False,
            include_color_table=False,
            disposal=0,
            lossless=False,
        )
