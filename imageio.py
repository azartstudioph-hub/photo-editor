"""Читання і запис фото: враховується орієнтація з EXIF та ручний поворот,
зберігаються колірний профіль і решта EXIF."""
import numpy as np
from PIL import Image, ImageOps

ORIENT = 274


def load(path, rotate=0, max_side=None):
    """Повертає (масив RGB, дані для запису). Орієнтація вже застосована."""
    im = Image.open(path)
    if max_side:
        im.draft('RGB', (max_side*2, max_side*2))
    # спершу розвертаємо пікселі за міткою EXIF: Pillow сам прибирає мітку
    im = ImageOps.exif_transpose(im)
    meta = dict(icc=im.info.get('icc_profile'), exif=im.info.get('exif'))
    im = im.convert('RGB')
    if max_side and max(im.size) > max_side:
        s = max_side/max(im.size)
        im = im.resize((max(int(im.width*s), 1), max(int(im.height*s), 1)), Image.LANCZOS)
    if rotate % 360:
        im = im.rotate(-(rotate % 360), expand=True)
    return np.asarray(im), meta


def save(arr, path, meta=None, quality=95):
    kw = dict(quality=quality, subsampling=0)
    if meta:
        if meta.get('icc'): kw['icc_profile'] = meta['icc']
        if meta.get('exif'): kw['exif'] = meta['exif']
    Image.fromarray(arr).save(path, **kw)
