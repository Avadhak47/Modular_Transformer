# Guide: Robust Training & Resuming on Kaggle (with Persistent Checkpoints)

This guide explains how to robustly train and resume your model on Kaggle, ensuring your checkpoints persist across sessions by using Kaggle Datasets. It also covers how to move checkpoints between sessions and directories.

---

## 1. Checkpoint Saving During Training
- **Checkpoints are saved in `TRANSFORMER/checkpoints/`** after each epoch.
- **Each checkpoint is uniquely named** with a timestamp, e.g., `kaggle_epoch_{epoch}_{timestamp}.pt`.

---

## 2. Persisting Checkpoints Across Sessions

### At the End of Each Epoch (Recommended)
- **Update your Kaggle Dataset with the latest checkpoints:**
  - Run the following command in a notebook cell:
    ```python
    !kaggle datasets version -p TRANSFORMER/checkpoints/ -m "Checkpoint after epoch {epoch}"
    ```
  - This will upload the latest checkpoints to your Dataset for persistence.
- **You cannot automate this at session end!** Sessions can end without warning, so update your Dataset frequently (ideally after every epoch).

### At the Start of a New Session
1. **Add your checkpoint dataset as an input:**
   - In the notebook sidebar, click "Add Data" and search for your dataset (e.g., `my-transformer-checkpoints`).
   - Add it to your notebook.
2. **Copy checkpoints from the input dataset to your workspace:**
   - Run the following command in a notebook cell:
     ```python
     !mkdir -p TRANSFORMER/checkpoints
     !cp -r /kaggle/input/my-transformer-checkpoints/checkpoints/* TRANSFORMER/checkpoints/
     ```
   - This ensures your script can find and resume from the latest checkpoint.

---

## 3. Resuming Training
- Use the `--resume` flag as before:
  ```bash
  python kaggle_train.py --pe_type sinusoidal --epochs 10 --batch_size 32 --model_size small --resume
  ```
- The script will automatically look for the latest checkpoint in `TRANSFORMER/checkpoints/` and resume from there.

---

## 4. Example: Full Workflow

**End of Each Epoch:**
- Save checkpoints (done automatically by script).
- Update your Kaggle Dataset with:
  ```python
  !kaggle datasets version -p TRANSFORMER/checkpoints/ -m "Checkpoint after epoch {epoch}"
  ```

**Start of New Session:**
- Add your checkpoint dataset as input.
- Copy checkpoints to your workspace:
  ```python
  !mkdir -p TRANSFORMER/checkpoints
  !cp -r /kaggle/input/my-transformer-checkpoints/checkpoints/* TRANSFORMER/checkpoints/
  ```
- Resume training with the script.

---

## 5. Tips
- **Always use unique names for your checkpoint datasets** (e.g., include the date or experiment name).
- **Checkpoints are named with timestamps** for uniqueness and to avoid overwriting.
- **You can copy only the latest checkpoint if storage is limited:**
  ```python
  !cp /kaggle/input/my-transformer-checkpoints/checkpoints/kaggle_epoch_9_20240717-153000.pt TRANSFORMER/checkpoints/
  ```
- **To list available checkpoints:**
  ```python
  !ls TRANSFORMER/checkpoints/
  ```
- **/kaggle/input/** is read-only. You cannot write to it; only copy from it.

---

## 6. Using the Kaggle API for Checkpoint Persistence
- To use the Kaggle API, you must have your Kaggle API credentials (`kaggle.json`) set up in your notebook.
- Example: After each epoch, run:
  ```python
  !kaggle datasets version -p TRANSFORMER/checkpoints/ -m "Checkpoint after epoch {epoch}"
  ```
- This will update your Dataset with the latest checkpoints.

---

## 7. Custom Dataset
- Replace the dummy dataset in the script with your actual dataset for real training.

---

Happy training & resuming, with persistent checkpoints! 