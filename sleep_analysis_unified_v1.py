import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yasa
import PyQt5
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input files
vhdr_file = r"C:\Users\ivales\Desktop\AMETS\1_Data\WP2\Pilot1\sub-P1001\ses-S001\eeg\SLEEP\WP2_Pilot1_P1001_S1_Oddball.vhdr"
output_dir = Path(r"C:\Users\ivales\Desktop\AMETS\3_Output\Pilot1\Analysis\Basic_Sleep\P1001")
output_dir.mkdir(exist_ok=True, parents=True)

# Analysis channels
eeg_channels = ['Cz', 'Pz']
eog_channel = 'EOG'
emg_channel = 'EMG'

# Parameters
confidence_threshold = 0.66
epoch_duration = 30
l_freq = 0.3
h_freq = 35.0

# File paths for manual annotations
manual_annotations_file = output_dir / 'annotations.csv'

# ============================================================================
# VISBRAIN COMPATIBILITY FIX
# ============================================================================

def patch_visbrain():
    """Patch VisBrain to fix PyQt compatibility issues."""
    try:
        import visbrain.gui.sleep.interface.ui_elements.ui_settings as ui_settings
        
        # Store original method
        original_fcn_slider_settings = ui_settings.UiSettings._fcn_slider_settings
        
        def patched_fcn_slider_settings(self):
            """Patched version that converts float to int for setTickInterval."""
            try:
                # Call original but catch the error
                original_fcn_slider_settings(self)
            except TypeError as e:
                if 'setTickInterval' in str(e):
                    # Fix the specific issue
                    step = self._SlVal.pageStep()
                    if isinstance(step, float):
                        self._SlVal.setTickInterval(int(step))
                    else:
                        self._SlVal.setTickInterval(step)
                else:
                    raise
        
        # Replace method
        ui_settings.UiSettings._fcn_slider_settings = patched_fcn_slider_settings
        print("  ✓ VisBrain compatibility patch applied")
        return True
        
    except Exception as e:
        print(f"  ⚠ Could not patch VisBrain: {e}")
        return False

# ============================================================================
# STAGE 1: AUTOMATIC STAGING WITH YASA
# ============================================================================

def stage_1_automatic_staging():
    """Perform automatic sleep staging with YASA and identify low confidence epochs."""
    
    print("="*80)
    print("STAGE 1: AUTOMATIC SLEEP STAGING WITH YASA")
    print("="*80)
    
    print("\n[1/6] Loading BrainVision data...")
    raw = mne.io.read_raw_brainvision(vhdr_file, preload=True)
    
    print(f"\nRecording information:")
    print(f"  Sampling frequency: {raw.info['sfreq']} Hz")
    print(f"  Duration: {raw.times[-1]/3600:.2f} hours")
    print(f"  Available channels: {raw.ch_names}")
    
    # Verify channels
    available_channels = []
    for ch in eeg_channels:
        if ch in raw.ch_names:
            available_channels.append(ch)
            print(f"✓ EEG channel found: {ch}")
        else:
            print(f"✗ EEG channel '{ch}' not found")
    
    if not available_channels:
        raise ValueError("No EEG channels available")
    
    eog_ch = eog_channel if eog_channel in raw.ch_names else None
    emg_ch = emg_channel if emg_channel in raw.ch_names else None
    
    if eog_ch:
        available_channels.append(eog_ch)
        print(f"✓ EOG channel found: {eog_ch}")
    if emg_ch:
        available_channels.append(emg_ch)
        print(f"✓ EMG channel found: {emg_ch}")
    
    # Preprocessing
    print("\n[2/6] Preprocessing signals...")
    raw_sleep = raw.copy().pick_channels(available_channels)
    
    channel_types = {ch: 'eeg' for ch in available_channels if ch in eeg_channels}
    if eog_ch:
        channel_types[eog_ch] = 'eog'
    if emg_ch:
        channel_types[emg_ch] = 'emg'
    
    raw_sleep.set_channel_types(channel_types)
    raw_sleep.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin')
    
    # Automatic staging
    print("\n[3/6] Running YASA automatic staging...")
    print("  This may take several minutes...")
    
    sls = yasa.SleepStaging(
        raw_sleep,
        eeg_name=available_channels[0],
        eog_name=eog_ch,
        emg_name=emg_ch
    )
    
    hypno_raw = sls.predict()
    proba = sls.predict_proba()
    
    print(f"\n  Staging completed: {len(hypno_raw)} epochs of {epoch_duration}s")
    
    # Convert to integers
    stage_names = {-1: 'Artefacto', 0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
    stage_to_int = {
        'Art': -1, 'Artefact': -1, 'ART': -1, 'Artifact': -1,
        'W': 0, 'Wake': 0, 'WAKE': 0, 'wake': 0,
        'N1': 1, '1': 1, 'n1': 1,
        'N2': 2, '2': 2, 'n2': 2,
        'N3': 3, '3': 3, 'n3': 3,
        'REM': 4, 'R': 4, 'rem': 4, 'Rem': 4
    }
    
    def convert_stage_to_int(stage):
        if isinstance(stage, (int, np.integer)):
            return int(stage)
        elif isinstance(stage, str):
            return stage_to_int.get(stage, 0)
        else:
            return int(stage)
    
    hypno = np.array([convert_stage_to_int(stage) for stage in hypno_raw], dtype=int)
    
    # Identify low confidence epochs
    print(f"\n[4/6] Identifying epochs with confidence < {confidence_threshold*100:.0f}%...")
    
    max_proba = proba.max(axis=1).values
    low_confidence_mask = max_proba < confidence_threshold
    n_low_confidence = np.sum(low_confidence_mask)
    
    print(f"  Low confidence epochs: {n_low_confidence} of {len(hypno)} ({n_low_confidence/len(hypno)*100:.1f}%)")
    
    hypno_corrected = hypno.copy().astype(np.int32)
    hypno_corrected[low_confidence_mask] = -1
    
    print(f"  Epochs marked as artifact: {np.sum(hypno_corrected == -1)}")
    
    # Create annotations
    print("\n[5/6] Creating annotations...")
    
    onset_times = []
    durations = []
    descriptions = []
    
    for i, stage in enumerate(hypno_corrected):
        onset = float(i * epoch_duration)
        stage_int = int(stage)
        stage_name = stage_names.get(stage_int, 'Unknown')
        
        onset_times.append(onset)
        durations.append(float(epoch_duration))
        descriptions.append(stage_name)
    
    # Save hypnogram with metadata
    visbrain_hypno = output_dir / 'visbrain_hypnogram.csv'
    visbrain_df = pd.DataFrame({
        'onset': onset_times,
        'duration': durations,
        'stage': hypno_corrected.astype(int),
        'description': descriptions,
        'confidence': max_proba,
        'needs_review': low_confidence_mask.astype(int)
    })
    visbrain_df.to_csv(visbrain_hypno, index=False)
    print(f"  ✓ Hypnogram saved: {visbrain_hypno}")
    
    # Save raw data WITHOUT annotations first to avoid the dtype error
    # Remove any existing annotations from the original file
    raw_sleep_clean = raw_sleep.copy()
    raw_sleep_clean.set_annotations(None)
    
    raw_sleep_path = output_dir / 'sleep_data.fif'
    raw_sleep_clean.save(raw_sleep_path, overwrite=True)
    print(f"  ✓ Raw sleep data saved: {raw_sleep_path}")
    
    # Create a simpler annotation object for VisBrain compatibility
    # We'll save annotations separately as CSV instead of embedding in FIF
    annotations_csv = output_dir / 'annotations_automatic.csv'
    pd.DataFrame({
        'onset': onset_times,
        'duration': durations,
        'description': descriptions
    }).to_csv(annotations_csv, index=False)
    print(f"  ✓ Annotations saved separately: {annotations_csv}")
    
    # Try to create annotated version with error handling
    try:
        # Use a more robust approach to create annotations
        annot = mne.Annotations(
            onset=onset_times, 
            duration=durations, 
            description=descriptions,
            orig_time=None  # Set to None to avoid potential datetime issues
        )
        
        raw_with_annotations = raw_sleep.copy()
        raw_with_annotations.set_annotations(annot)
        
        # Try saving with split files to avoid large file issues
        raw_fif_path = output_dir / 'sleep_data_with_staging.fif'
        raw_with_annotations.save(raw_fif_path, overwrite=True, split_naming='neuromag')
        print(f"  ✓ Annotated data saved: {raw_fif_path}")
        
    except Exception as e:
        print(f"  ⚠ Could not save annotated FIF file: {e}")
        print(f"  → Using annotation CSV file instead")
        raw_with_annotations = raw_sleep.copy()
        # Add annotations in memory only
        annot = mne.Annotations(
            onset=onset_times, 
            duration=durations, 
            description=descriptions,
            orig_time=None
        )
        raw_with_annotations.set_annotations(annot)
    
    print("\n[6/6] Generating visualizations...")
    
    time_hours = np.arange(len(hypno)) * epoch_duration / 3600
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    
    ax1 = axes[0]
    ax1.plot(time_hours, hypno_corrected, 'b-', linewidth=1.5)
    ax1.scatter(time_hours[low_confidence_mask], hypno_corrected[low_confidence_mask], 
                c='red', s=15, alpha=0.7, label=f'Low confidence (<{confidence_threshold*100:.0f}%)')
    ax1.set_ylabel('Sleep stage', fontsize=11)
    ax1.set_yticks([-1, 0, 1, 2, 3, 4])
    ax1.set_yticklabels(['ART', 'W', 'N1', 'N2', 'N3', 'REM'])
    ax1.set_title(f'Hypnogram - Epochs with confidence <{confidence_threshold*100:.0f}% marked as artifacts', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.plot(time_hours, max_proba, 'k-', linewidth=1, alpha=0.7)
    ax2.axhline(confidence_threshold, color='r', linestyle='--', linewidth=2, 
                label=f'Threshold: {confidence_threshold*100:.0f}%')
    ax2.fill_between(time_hours, 0, 1, where=low_confidence_mask, 
                      color='red', alpha=0.2, label='Low confidence')
    ax2.set_xlabel('Time (hours)', fontsize=11)
    ax2.set_ylabel('Maximum confidence', fontsize=11)
    ax2.set_title('Automatic Staging Confidence', fontsize=12)
    ax2.set_ylim([0, 1])
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = output_dir / 'hypnogram_with_confidence.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Hypnogram saved: {fig_path}")
    
    print("\n" + "="*80)
    print("STAGE 1 COMPLETED")
    print("="*80)
    print(f"\nEpochs requiring manual review: {n_low_confidence} ({n_low_confidence/len(hypno)*100:.1f}%)")
    print(f"Files saved in: {output_dir.absolute()}")
    
    return raw_with_annotations, hypno_corrected, visbrain_df

# ============================================================================
# STAGE 2: MANUAL CORRECTION WITH VISBRAIN
# ============================================================================

def stage_2_manual_correction(raw_with_annotations, hypno_corrected):
    """Launch VisBrain for manual correction of artifacts."""
    
    print("\n" + "="*80)
    print("STAGE 2: MANUAL CORRECTION WITH VISBRAIN")
    print("="*80)
    
    print("\nPreparing VisBrain interface...")
    
    # Apply compatibility patch
    patch_success = patch_visbrain()
    
    # Expand hypnogram to samples
    sf = raw_with_annotations.info['sfreq']
    n_samples = raw_with_annotations.n_times
    hypno_expanded = np.zeros(n_samples, dtype=int)
    
    for i, stage in enumerate(hypno_corrected):
        start_sample = int(i * epoch_duration * sf)
        end_sample = int((i + 1) * epoch_duration * sf)
        end_sample = min(end_sample, n_samples)
        hypno_expanded[start_sample:end_sample] = int(stage)
    
    print(f"Expanded hypnogram: {len(hypno_expanded)} samples")
    print(f"Raw data: {n_samples} samples")
    
    print("\n" + "="*80)
    print("LAUNCHING VISBRAIN")
    print("="*80)
    print("\nINSTRUCTIONS:")
    print("1. Review and correct the artifacts (red epochs)")
    print("2. Make your manual annotations")
    print("3. Export annotations using: File > Save scoring")
    print(f"4. Save as: {manual_annotations_file}")
    print("5. Close VisBrain when finished")
    print("\nPress any key in the console after closing VisBrain to continue...")
    print("="*80 + "\n")
    
    # Launch VisBrain with error handling
    try:
        from visbrain.gui import Sleep
        Sleep(data=raw_with_annotations.get_data(), 
              sf=sf,
              channels=raw_with_annotations.ch_names,
              hypno=hypno_expanded).show()
        
        print("\nVisBrain closed.")
        
    except Exception as e:
        print(f"\n✗ Error launching VisBrain: {e}")
        print("\nTroubleshooting options:")
        print("1. Try downgrading PyQt5: pip install PyQt5==5.14.0")
        print("2. Use MNE's built-in viewer instead")
        print("3. Skip manual correction and use automatic staging only")
        
        choice = input("\nDo you want to skip manual correction and continue? (y/n): ")
        if choice.lower() != 'y':
            raise
        else:
            print("Skipping manual correction stage...")
            return
    
    input("Press ENTER to continue with Stage 3 (merging annotations)...")

# ============================================================================
# STAGE 3: MERGE AUTOMATIC AND MANUAL ANNOTATIONS
# ============================================================================

def stage_3_merge_annotations():
    """Merge manual corrections with automatic staging results."""
    
    print("\n" + "="*80)
    print("STAGE 3: MERGING ANNOTATIONS")
    print("="*80)
    
    print("\n[1/4] Loading automatic hypnogram...")
    hypno_auto = pd.read_csv(output_dir / 'visbrain_hypnogram.csv')
    print(f"  ✓ Loaded {len(hypno_auto)} epochs")
    
    print("\n[2/4] Loading manual annotations...")
    if not manual_annotations_file.exists():
        print(f"  ⚠ Manual annotations file not found: {manual_annotations_file}")
        print("  Using only automatic staging.")
        hypno_final = hypno_auto.copy()
    else:
        hypno_manual = pd.read_csv(manual_annotations_file, 
                                   header=None,
                                   names=['onset', 'end', 'description'])
        
        hypno_manual = hypno_manual.dropna()
        hypno_manual['description'] = hypno_manual['description'].str.strip()
        
        # Normalize stage names
        hypno_manual['description'] = hypno_manual['description'].replace({
            'Waje': 'Wake',
            'waje': 'Wake',
            'wake': 'Wake',
            'WAKE': 'Wake'
        })
        
        print(f"  ✓ Loaded {len(hypno_manual)} manual annotations")
        
        # Map stages
        stage_map = {'Wake': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'REM': 4, 'Artefacto': -1, 'Art': -1}
        hypno_manual['stage'] = hypno_manual['description'].map(stage_map)
        
        # Create dictionary for fast lookup
        manual_dict = {int(row['onset']): row for _, row in hypno_manual.iterrows()}
        
        print("\n[3/4] Merging annotations...")
        hypno_final = hypno_auto.copy()
        replacements = 0
        
        for idx, row in hypno_final.iterrows():
            onset_key = int(row['onset'])
            if row['description'] == 'Artefacto' and onset_key in manual_dict:
                manual = manual_dict[onset_key]
                hypno_final.loc[idx, 'stage'] = manual['stage']
                hypno_final.loc[idx, 'description'] = manual['description']
                hypno_final.loc[idx, 'needs_review'] = 0
                replacements += 1
        
        print(f"  ✓ Replaced {replacements} artifacts with manual annotations")
    
    print("\n[4/4] Saving final hypnogram...")
    output_path = output_dir / 'hypnogram_final.csv'
    hypno_final.to_csv(output_path, index=False)
    print(f"  ✓ Final hypnogram saved: {output_path}")
    
    print("\nFinal stage distribution:")
    print(hypno_final['description'].value_counts())
    
    # Update FIF file with corrected annotations - with better error handling
    print("\nUpdating FIF file with corrected annotations...")
    try:
        raw = mne.io.read_raw_fif(output_dir / 'sleep_data.fif', preload=True)
        
        onset_times = []
        durations = []
        descriptions = []
        
        for idx, row in hypno_final.iterrows():
            onset_times.append(float(row['onset']))
            durations.append(float(row['duration']))
            descriptions.append(str(row['description']))
        
        annot = mne.Annotations(
            onset=onset_times, 
            duration=durations, 
            description=descriptions,
            orig_time=None
        )
        
        raw_corrected = raw.copy()
        raw_corrected.set_annotations(annot)
        
        corrected_path = output_dir / 'sleep_data_corrected.fif'
        raw_corrected.save(corrected_path, overwrite=True, split_naming='neuromag')
        print(f"  ✓ Corrected data saved: {corrected_path}")
        
    except Exception as e:
        print(f"  ⚠ Could not save corrected FIF with annotations: {e}")
        print(f"  → Annotations are available in: {output_path}")
        print(f"  → You can load raw data from: {output_dir / 'sleep_data.fif'}")
        print(f"  → And apply annotations from the CSV file manually")
    
    print("\n" + "="*80)
    print("STAGE 3 COMPLETED")
    print("="*80)
    
    return hypno_final

# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """Execute complete sleep analysis workflow."""
    
    print("\n" + "="*80)
    print("INTEGRATED SLEEP ANALYSIS WORKFLOW")
    print("="*80)
    print("\nThis workflow consists of three stages:")
    print("  1. Automatic staging with YASA (confidence threshold: 66%)")
    print("  2. Manual correction with VisBrain")
    print("  3. Merge automatic and manual annotations")
    print("\n" + "="*80 + "\n")
    
    try:
        # Stage 1: Automatic staging
        raw_with_annotations, hypno_corrected, visbrain_df = stage_1_automatic_staging()
        
        # Stage 2: Manual correction
        stage_2_manual_correction(raw_with_annotations, hypno_corrected)
        
        # Stage 3: Merge annotations
        hypno_final = stage_3_merge_annotations()
        
        print("\n" + "="*80)
        print("WORKFLOW COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"\nAll files saved in: {output_dir.absolute()}")
        print("\nKey output files:")
        print(f"  - hypnogram_final.csv: Final merged hypnogram")
        print(f"  - visbrain_hypnogram.csv: Original automatic staging")
        print(f"  - sleep_data.fif: Raw sleep data (without annotations)")
        print(f"  - annotations_automatic.csv: Automatic annotations")
        print("\nNote: If FIF files with embedded annotations failed to save,")
        print("you can load the raw FIF and apply annotations from CSV files.")
        
    except Exception as e:
        print(f"\n✗ Error in workflow: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()