@echo on

pyinstaller --noconfirm ^
--log-level=WARN ^
--windowed ^
--onedir ^
--icon=app_icon.ico ^
stim_tool.spec