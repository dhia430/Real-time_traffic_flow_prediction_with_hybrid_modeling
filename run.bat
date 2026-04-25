@echo off
set PYTHONPATH=.
echo ===========================================
echo Real-Time Traffic Flow Prediction Pipeline
echo ===========================================

REM Ensure dependencies are installed
echo Checking dependencies...
pip install -r requirements.txt > nul 2>&1

:menu
echo.
echo Please select an option:
echo 1) Prepare Dataset (Extract and Split)
echo 2) Train YOLO Model
echo 3) Run Pipeline on Video
echo 4) Run Tests (CTM, Density, etc.)
echo 5) Run Verification Scripts
echo 6) Run Pipeline on Live Camera
echo.
set /p choice="Enter choice (1-6): "

if "%choice%"=="1" goto prepare_dataset
if "%choice%"=="2" goto train_yolo
if "%choice%"=="3" goto run_pipeline
if "%choice%"=="4" goto run_tests
if "%choice%"=="5" goto run_scripts
if "%choice%"=="6" goto run_live

echo Invalid choice. Exiting.
goto end

:prepare_dataset
echo.
echo Do you want to process a new data set or use the existing one?
echo 1) New single dataset (.zip)
echo 2) Merge TWO datasets (.zip) for best performance
echo 3) Existing dataset (re-split)
set /p data_choice="Enter choice (1-3): "

if "%data_choice%"=="1" goto prepare_new
if "%data_choice%"=="2" goto prepare_merge
goto prepare_existing

:prepare_new
set /p new_data_path="Enter path to the new data (.zip): "
set "new_data_path=%new_data_path:"=%"
echo Preparing new dataset from: %new_data_path%
python src\utils\data_splitter.py --raw_zip "%new_data_path%"
echo.
echo Dataset prepared! A new timestamped folder has been created in data\.
echo You can now select option 2 with Fine-tuning to train on this new data.
goto prompt_continue

:prepare_merge
set /p zip1_path="Enter path to FIRST dataset (.zip): "
set "zip1_path=%zip1_path:"=%"
set /p zip2_path="Enter path to SECOND dataset (.zip): "
set "zip2_path=%zip2_path:"=%"
echo Merging both datasets...
python src\utils\data_splitter.py --raw_zip "%zip1_path%" --raw_zip2 "%zip2_path%"
echo.
echo Merged dataset prepared! Select option 2 with Fine-tuning to train on it.
goto prompt_continue

:prepare_existing
echo Preparing existing dataset...
python src\utils\data_splitter.py
goto prompt_continue

:train_yolo
echo.
echo Do you want to train on a new data set (fine-tune) or train on the existing one?
echo 1) Fine-tune on new dataset (uses previous best.pt)
echo 2) Train on existing dataset
set /p train_choice="Enter choice (1-2): "

if "%train_choice%"=="1" goto train_finetune
goto train_existing

:train_finetune
echo.
echo Checking for new dataset prepared in step 1...
if exist "data\.last_new_data_yaml.txt" (
    set /p NEW_DATA_YAML=<"data\.last_new_data_yaml.txt"
    echo Found new dataset yaml: %NEW_DATA_YAML%
    echo Fine-tuning on NEW data using best.pt weights...
    python src\cv\train_yolo.py --config config\config.yaml --weights models\saved_models\weights\best.pt --data_yaml "%NEW_DATA_YAML%"
) else (
    echo WARNING: No new dataset found from step 1.
    echo Please run option 1 with a new dataset zip first, then retry fine-tuning.
    echo Falling back to existing dataset fine-tuning...
    python src\cv\train_yolo.py --config config\config.yaml --weights models\saved_models\weights\best.pt
)
goto prompt_continue

:train_existing
echo Training YOLO model...
python src\cv\train_yolo.py --config config\config.yaml
goto prompt_continue

:run_pipeline
echo Running full pipeline...
set /p video_path="Enter path to video file: "
set "video_path=%video_path:"=%"
python src\pipeline\run_pipeline.py --video "%video_path%" --config config\config.yaml --output outputs\result.mp4
goto prompt_continue

:run_live
echo Running pipeline on Live Camera...
python src\pipeline\run_pipeline.py --video 0 --config config\config.yaml --output outputs\result_live.mp4
goto prompt_continue

:run_tests
echo Running unit tests...
pytest tests\test_models.py -v
goto prompt_continue

:run_scripts
echo Running verification scripts...
python verify_detection.py
python verify_density.py
python verify_ctm.py
goto prompt_continue

:prompt_continue
echo.
set /p continue_choice="Do you want to do another step? (y/n): "
if /i "%continue_choice%"=="y" goto menu
if /i "%continue_choice%"=="yes" goto menu

:end
pause
