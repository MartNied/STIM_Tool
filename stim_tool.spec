# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(['stim_tool.py'],
             pathex=['D:\\programming\\python\\STIM_Tool'],
             binaries=[],
             datas=[("stim_tool_main_window.ui", ".")],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='stim_tool',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False,
          icon="D:\\programming\\python\\STIM_Tool\\icons\\app_icon.ico")


a.datas += Tree('./icons', prefix='icons')

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='stim_tool')



