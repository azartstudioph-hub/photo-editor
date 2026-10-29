# Збірка самостійного застосунку: pyinstaller azart.spec
# Результат: dist/AZART/AZART.exe — Python на комп'ютері не потрібен.
block_cipher = None

a = Analysis(
    ['app.py'],
    pathex=['.'],
    binaries=[],
    datas=[('ui.html', '.'), ('models/yunet2.onnx', 'models'), ('README.md', '.')],
    hiddenimports=['enhance', 'runner', 'imageio'],
    hookspath=[], runtime_hooks=[],
    excludes=['matplotlib', 'tkinter', 'scipy', 'skimage', 'pandas', 'IPython'],
    cipher=block_cipher, noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='AZART',
          debug=False, strip=False, upx=False, console=True)
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, strip=False, upx=False, name='AZART')
