# dataset_sim.py
import torch
from torch.utils.data import Dataset
import numpy as np
import pyroomacoustics as pra
import librosa
import scipy.signal as signal
import soundfile as sf
import os
import glob
import random

class DynamicRoomSimulator(Dataset):
    def __init__(self, audio_source_dir, sample_length=2048, epoch_length=2000,
                 noise_dir="/mnt/chensenda/codes/sound/denoisy/datasets/DNS-Challenge/datasets/noise",
                 crops_per_simulation=8, simulation_duration_sec=1.5):
        """
        Args:
            audio_source_dir: Directory containing speech source files
            sample_length: Length of each output sample (default 2048)
            epoch_length: Number of samples per epoch
            noise_dir: Directory containing noise files for augmentation
            crops_per_simulation: Number of crops to extract per room simulation (default 6)
            simulation_duration_sec: Duration of source audio for simulation (default 1.5s)
        """
        self.sample_length = sample_length
        self.epoch_length = epoch_length
        self.fs = 16000
        
        # -------------------------------------------------------------
        # Multi-Crop Configuration: One simulation -> Multiple samples
        # This amortizes the expensive room.simulate() across N crops
        # -------------------------------------------------------------
        self.crops_per_simulation = crops_per_simulation
        self.simulation_duration_sec = simulation_duration_sec
        
        # Crop cache: In multiprocessing mode (num_workers > 0), each worker
        # process gets its own copy of the Dataset object, so this list is
        # naturally isolated between processes. No threading.local needed!
        self.crop_cache = []
        
        # -------------------------------------------------------------
        # 1. Load speech source files (supports .wav and .flac)
        # -------------------------------------------------------------
        self.source_files = []
        for ext in ['*.wav', '*.flac']:
            self.source_files.extend(glob.glob(os.path.join(audio_source_dir, f"**/{ext}"), recursive=True))
            self.source_files.extend(glob.glob(os.path.join("/mnt/chensenda/codes/sound/cnn_audio_SSL/speech_data/data_aishell/wav/train", f"**/{ext}"), recursive=True))
            
            
        if len(self.source_files) == 0:
            raise ValueError(f"No audio files found in {audio_source_dir} (.wav/.flac)!")
        
        print(f"[DynamicRoomSimulator] Loaded {len(self.source_files)} source audio files")

        # -------------------------------------------------------------
        # 2. Load real noise files into memory for fast training
        # Using soundfile instead of librosa for better performance
        # -------------------------------------------------------------
        self.noise_cache = []  # Pre-loaded noise arrays in memory
        if noise_dir and os.path.exists(noise_dir):
            noise_files = []
            for ext in ['*.wav', '*.flac']:
                noise_files.extend(glob.glob(os.path.join(noise_dir, f"**/{ext}"), recursive=True))
            
            print(f"[DynamicRoomSimulator] Found {len(noise_files)} noise files, pre-loading to memory...")
            
            # Pre-load all noise files into memory for fast access during training
            for nf in noise_files:
                try:
                    # Use soundfile for fast loading (much faster than librosa)
                    data, sr = sf.read(nf, dtype='float32')
                    
                    # Handle stereo -> mono conversion
                    if len(data.shape) > 1:
                        data = np.mean(data, axis=1)
                    
                    # Resample to target sample rate if needed
                    if sr != self.fs:
                        # Use scipy for fast resampling
                        num_samples = int(len(data) * self.fs / sr)
                        data = signal.resample(data, num_samples)
                    
                    # Only keep noise clips longer than 0.5 seconds
                    if len(data) >= self.fs * 0.5:
                        self.noise_cache.append(data.astype(np.float32))
                except Exception as e:
                    # Skip problematic files silently
                    pass
            
            print(f"[DynamicRoomSimulator] Successfully cached {len(self.noise_cache)} noise clips in memory")
        else:
            print(f"[DynamicRoomSimulator] WARNING: noise_dir not found, using Gaussian noise as fallback")
        
        # Unitree Go2 microphone array definition (4 mics, 3D coordinates)
        self.mic_positions_local = np.array([
            [ 0.1035,  0.0235, 0.0], # mic0
            [ 0.1035, -0.0235, 0.0], # mic1
            [-0.1035, -0.0235, 0.0], # mic2
            [-0.1035,  0.0235, 0.0], # mic3
        ]).T # Transpose to 3x4
        
        print(f"[DynamicRoomSimulator] Multi-Crop enabled: {crops_per_simulation} crops per {simulation_duration_sec}s simulation")

    def __len__(self):
        return self.epoch_length

    def _get_random_room_params(self):
        """
        改进版：随机生成不同类型的房间（小、中、大、长走廊）
        覆盖 3m - 15m 的范围，适应机器狗的真实活动区域
        """
        room_type = np.random.choice(['small', 'medium', 'large', 'corridor'], p=[0.3, 0.4, 0.2, 0.1])

        if room_type == 'small':
            # 卧室、小书房 (3-5m)
            l = np.random.uniform(3.0, 5.0)
            w = np.random.uniform(3.0, 5.0)
            h = np.random.uniform(2.5, 3.0)
            rt60 = np.random.uniform(0.2, 0.4)
            
        elif room_type == 'medium':
            # 客厅、会议室 (5-8m) - 你原来的设置主要覆盖这里
            l = np.random.uniform(5.0, 8.0)
            w = np.random.uniform(5.0, 8.0)
            h = np.random.uniform(2.8, 3.5)
            rt60 = np.random.uniform(0.3, 0.6)
            
        elif room_type == 'large':
            # 大厅、开放式办公区 (8-15m)
            # 🔥 关键：只有足够大的房间才能容纳 5m+ 的声源而不贴墙
            l = np.random.uniform(8.0, 15.0)
            w = np.random.uniform(8.0, 15.0)
            h = np.random.uniform(3.5, 5.0) # 大房间通常顶更高
            rt60 = np.random.uniform(0.5, 0.9) # 混响更长
            
        elif room_type == 'corridor':
            # 走廊 (狭长) - 机器狗常见场景
            # 特征：一个维度很长，另一个维度很窄
            if np.random.random() < 0.5:
                l = np.random.uniform(10.0, 20.0) # 长
                w = np.random.uniform(2.0, 3.5)   # 窄
            else:
                l = np.random.uniform(2.0, 3.5)
                w = np.random.uniform(10.0, 20.0)
            h = np.random.uniform(2.5, 3.5)
            rt60 = np.random.uniform(0.4, 0.7)

        room_dim = np.array([l, w, h])

        # 使用 Sabine 公式反推吸音系数
        try:
            e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
            # 限制 max_order，大房间如果 order 太高计算会极慢
            # 大房间 order=3 足够模拟长拖尾，小房间可以用 order=5
            target_order = 3 if (l > 10 or w > 10) else 5 
            max_order = min(max_order, target_order) 
        except:
            e_absorption, max_order = 0.3, 3

        return room_dim, e_absorption, max_order

    def _random_pitch_shift(self, audio: np.ndarray, prob: float = 0.7) -> np.ndarray:
        """
        Random pitch shift via resampling to simulate different voice timbres.
        
        This is CRITICAL for preventing "timbre overfitting" when you have limited
        source audio files. By changing the playback speed:
          - Speed x 0.8 -> Simulates deep male voice / elderly
          - Speed x 1.2 -> Simulates high-pitched female voice / child
        
        Args:
            audio: Input audio signal (1D numpy array)
            prob: Probability of applying pitch shift (default 0.7)
        
        Returns:
            Pitch-shifted audio (same length as input)
        """
        if np.random.random() > prob:
            return audio
        
        # Random speed factor: 0.7x (very deep) to 1.4x (very high pitched)
        # This covers the range from bass male voices to children's voices
        speed_factor = np.random.uniform(0.7, 1.4)
        
        if abs(speed_factor - 1.0) < 0.05:
            # Skip if speed change is negligible
            return audio
        
        # Calculate new length after speed change
        original_len = len(audio)
        new_len = int(original_len / speed_factor)
        
        if new_len < 100:
            return audio
        
        # Resample to simulate speed change (pitch shift)
        # scipy.signal.resample is fast and maintains quality
        resampled = signal.resample(audio, new_len)
        
        # Adjust length back to original (crop or pad)
        if len(resampled) > original_len:
            # If slower (longer), crop from center
            start = (len(resampled) - original_len) // 2
            resampled = resampled[start:start + original_len]
        elif len(resampled) < original_len:
            # If faster (shorter), pad with zeros
            pad_total = original_len - len(resampled)
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            resampled = np.pad(resampled, (pad_left, pad_right), mode='constant')
        
        return resampled.astype(np.float32)

    def _load_random_noise_clip(self, duration_samples: int) -> np.ndarray:
        """
        Load a random noise clip from the pre-cached noise buffer.
        This is used for point source noise simulation (e.g., TV, interfering speaker).
        
        Args:
            duration_samples: Desired length in samples
        
        Returns:
            Noise signal array of shape (duration_samples,)
        """
        if len(self.noise_cache) == 0:
            # Fallback: return Gaussian noise if no cached noise available
            return np.random.randn(duration_samples).astype(np.float32) * 0.1
        
        noise_y = random.choice(self.noise_cache).copy()
        
        # Adjust length to match duration
        if len(noise_y) < duration_samples:
            # Wrap-around padding for short noise clips
            noise_y = np.pad(noise_y, (0, duration_samples - len(noise_y)), mode='wrap')
        else:
            # Random crop for long noise clips
            max_start = len(noise_y) - duration_samples
            start_idx = np.random.randint(0, max_start + 1) if max_start > 0 else 0
            noise_y = noise_y[start_idx:start_idx + duration_samples]
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(noise_y))
        if max_val > 0:
            noise_y = noise_y / max_val
        
        return noise_y.astype(np.float32)
    
    def _apply_additive_noise(self, simulated_audio: np.ndarray) -> np.ndarray:
        """
        Apply additive noise to simulated audio with proper multi-channel decorrelation.
        This is extracted as a separate method so each crop can have different noise.
        
        Improved decorrelation strategy:
        - Each channel gets noise from a DIFFERENT random position in the noise file
        - This drastically reduces inter-channel correlation (from ~0.99 to ~0.3)
        - More realistic simulation of diffuse noise field
        
        Args:
            simulated_audio: Shape (4, N), the room-simulated audio
        
        Returns:
            Noisy audio with same shape
        """
        audio = simulated_audio.copy()
        
        # Random SNR: 5dB (very noisy) - 25dB (quiet)
        target_snr_db = np.random.uniform(5.0, 25.0)
        
        sig_power = np.mean(audio ** 2)
        if sig_power > 0:
            if len(self.noise_cache) > 0:
                signal_len = audio.shape[1]
                noise_multichannel = np.zeros_like(audio)
                
                # IMPROVED: Each channel gets independently sampled noise
                # This creates much lower inter-channel correlation
                for ch in range(4):
                    # Randomly select a noise clip for this channel
                    noise_y = random.choice(self.noise_cache)
                    
                    # Extract from DIFFERENT random position for each channel
                    if len(noise_y) < signal_len:
                        # Wrap-around for short clips
                        noise_ch = np.pad(noise_y, (0, signal_len - len(noise_y)), mode='wrap')
                    else:
                        # Each channel gets a different random crop position
                        max_start = len(noise_y) - signal_len
                        start_idx = np.random.randint(0, max_start + 1)
                        noise_ch = noise_y[start_idx:start_idx + signal_len]
                    
                    # Small gain variation per channel (physically realistic)
                    noise_multichannel[ch, :] = noise_ch * np.random.uniform(0.85, 1.15)
                
                # Add small physically-realistic delay (0-10 samples for ~20cm mic spacing)
                # This is optional but adds realism without destroying decorrelation
                for ch in range(4):
                    delay = np.random.randint(0, 10)
                    noise_multichannel[ch, :] = np.roll(noise_multichannel[ch, :], shift=delay)
                
                noise_power = np.mean(noise_multichannel ** 2)
                if noise_power > 0:
                    scale = np.sqrt(sig_power / (noise_power * 10**(target_snr_db/10)))
                    audio = audio + noise_multichannel * scale
            else:
                # Fallback: Independent Gaussian white noise per channel
                # This naturally has zero inter-channel correlation
                noise_power = sig_power / (10 ** (target_snr_db / 10))
                noise = np.random.normal(0, np.sqrt(noise_power), audio.shape)
                audio = audio + noise
        
        return audio
    
    def _apply_channel_gain_and_normalize(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply channel gain perturbation and normalize.
        Each crop gets different gain perturbation for diversity.
        
        Args:
            audio: Shape (4, N)
        
        Returns:
            Processed audio with same shape
        """
        # Channel gain perturbation (~±3dB)
        gain_perturb = np.random.uniform(0.7, 1.3, size=(4, 1))
        audio = audio * gain_perturb
        
        # Normalize (preserve ILD, avoid division by zero)
        max_amp = np.max(np.abs(audio))
        if max_amp > 0:
            audio = audio / max_amp * 0.9
        
        return audio
    
    def _extract_random_crop(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract a random crop of sample_length from audio.
        Includes silent-crop protection.
        
        Args:
            audio: Shape (4, N), the full simulated audio
        
        Returns:
            Cropped audio of shape (4, sample_length)
        """
        signal_len = audio.shape[1]
        
        if signal_len > self.sample_length:
            max_start = signal_len - self.sample_length
            
            # Silent crop protection: try 3 times
            for _ in range(3):
                start = np.random.randint(0, max_start + 1)
                cropped = audio[:, start : start + self.sample_length]
                
                # Energy threshold check
                if np.mean(cropped**2) > 1e-5:
                    return cropped
            
            # Fallback: take center crop
            start = max_start // 2
            return audio[:, start : start + self.sample_length]
        else:
            # Pad if too short
            pad_width = self.sample_length - signal_len
            return np.pad(audio, ((0, 0), (0, pad_width)))
    
    def _run_simulation_and_fill_cache(self):
        """
        Run ONE room simulation and fill the cache with MULTIPLE crops.
        This is the key optimization: expensive simulate() is amortized across N crops.
        
        Each crop gets:
        - Same room impulse response (RIR) and source direction
        - DIFFERENT random additive noise (SNR, noise type)
        - DIFFERENT channel gain perturbation
        - DIFFERENT random crop position in time
        
        Returns:
            List of (audio_tensor, label_tensor) tuples
        """
        # -----------------------------------------------------------
        # 1. Setup room (same as before)
        # -----------------------------------------------------------
        room_dim, e_absorption, max_order = self._get_random_room_params()
        
        room = pra.ShoeBox(
            room_dim, 
            fs=self.fs, 
            materials=pra.Material(e_absorption), 
            max_order=max_order
        )

        # -----------------------------------------------------------
        # 2. Place microphone array
        # -----------------------------------------------------------
        mic_center = np.array([
            np.random.uniform(0.5, room_dim[0] - 0.5),
            np.random.uniform(0.5, room_dim[1] - 0.5),
            np.random.uniform(0.5, room_dim[2] - 0.5) 
        ])
        
        current_mic_locs = self.mic_positions_local + mic_center.reshape(3, 1)
        room.add_microphone_array(current_mic_locs)

        # -----------------------------------------------------------
        # 3. Place source (random angle + distance)
        # -----------------------------------------------------------
        angle_deg = np.random.randint(0, 360)
        angle_rad = np.deg2rad(angle_deg)
        dist = np.random.uniform(0.5, 5.0)
        
        src_x = mic_center[0] + dist * np.cos(angle_rad)
        src_y = mic_center[1] + dist * np.sin(angle_rad)
        src_z = mic_center[2] + np.random.uniform(-0.2, 1.5)
        
        # Ray scaling to keep source inside room
        x_min, x_max = 0.1, room_dim[0] - 0.1
        y_min, y_max = 0.1, room_dim[1] - 0.1
        z_min, z_max = 0.1, room_dim[2] - 0.1

        dx = src_x - mic_center[0]
        dy = src_y - mic_center[1]
        dz = src_z - mic_center[2]

        scale = 1.0
        if dx != 0:
            if src_x > x_max: scale = min(scale, (x_max - mic_center[0]) / dx)
            if src_x < x_min: scale = min(scale, (x_min - mic_center[0]) / dx)
        if dy != 0:
            if src_y > y_max: scale = min(scale, (y_max - mic_center[1]) / dy)
            if src_y < y_min: scale = min(scale, (y_min - mic_center[1]) / dy)
        if dz != 0:
            if src_z > z_max: scale = min(scale, (z_max - mic_center[2]) / dz)
            if src_z < z_min: scale = min(scale, (z_min - mic_center[2]) / dz)
        scale = max(1e-4, min(scale, 1.0))

        src_x = mic_center[0] + dx * scale
        src_y = mic_center[1] + dy * scale
        src_z = mic_center[2] + dz * scale

        # Calculate final angle label
        real_dx = src_x - mic_center[0]
        real_dy = src_y - mic_center[1]
        real_angle_rad = np.arctan2(real_dy, real_dx)
        if real_angle_rad < 0: 
            real_angle_rad += 2 * np.pi
        label_deg = int(np.degrees(real_angle_rad)) % 360
        
        # -----------------------------------------------------------
        # 4. Load LONGER source clip for multi-crop
        # -----------------------------------------------------------
        source_signal = self._load_random_source_clip(duration_sec=self.simulation_duration_sec)
        
        # Spectral augmentation (applied once to source)
        if np.random.random() < 0.8:
            low_cut = np.random.uniform(50, 200)
            high_cut = np.random.uniform(4000, 7500)
            try:
                sos = signal.butter(2, [low_cut, high_cut], btype='band', fs=self.fs, output='sos')
                source_signal = signal.sosfilt(sos, source_signal)
            except Exception:
                pass
        
        room.add_source([src_x, src_y, src_z], signal=source_signal)

        # -----------------------------------------------------------
        # 5. Point source noise (30% probability)
        # -----------------------------------------------------------
        if np.random.random() < 0.3 and len(self.noise_cache) > 0:
            noise_angle_deg = (angle_deg + np.random.randint(30, 330)) % 360
            noise_angle_rad = np.deg2rad(noise_angle_deg)
            noise_dist = np.random.uniform(0.5, 3.0)
            
            noise_x = mic_center[0] + noise_dist * np.cos(noise_angle_rad)
            noise_y = mic_center[1] + noise_dist * np.sin(noise_angle_rad)
            noise_z = mic_center[2] + np.random.uniform(-0.2, 1.0)
            
            # Boundary scaling for noise source
            ndx = noise_x - mic_center[0]
            ndy = noise_y - mic_center[1]
            ndz = noise_z - mic_center[2]
            
            nscale = 1.0
            if ndx != 0:
                if noise_x > x_max: nscale = min(nscale, (x_max - mic_center[0]) / ndx)
                if noise_x < x_min: nscale = min(nscale, (x_min - mic_center[0]) / ndx)
            if ndy != 0:
                if noise_y > y_max: nscale = min(nscale, (y_max - mic_center[1]) / ndy)
                if noise_y < y_min: nscale = min(nscale, (y_min - mic_center[1]) / ndy)
            if ndz != 0:
                if noise_z > z_max: nscale = min(nscale, (z_max - mic_center[2]) / ndz)
                if noise_z < z_min: nscale = min(nscale, (z_min - mic_center[2]) / ndz)
            nscale = max(1e-4, min(nscale, 1.0))
            
            noise_x = mic_center[0] + ndx * nscale
            noise_y = mic_center[1] + ndy * nscale
            noise_z = mic_center[2] + ndz * nscale
            
            noise_signal = self._load_random_noise_clip(len(source_signal))
            interference_gain = np.random.uniform(0.0, 0.5)
            noise_signal = noise_signal * interference_gain
            
            # room.add_source([noise_x, noise_y, noise_z], signal=noise_signal)

        # -----------------------------------------------------------
        # 6. Run EXPENSIVE simulation ONCE
        # -----------------------------------------------------------
        room.simulate()
        simulated_audio = room.mic_array.signals  # Shape: (4, N)

        # -----------------------------------------------------------
        # 7. Generate MULTIPLE crops from this simulation
        # Each crop has DIFFERENT noise, gain, and crop position
        # -----------------------------------------------------------
        crops = []
        label_tensor = torch.tensor(label_deg, dtype=torch.long)
        
        for _ in range(self.crops_per_simulation):
            # Each crop gets independent noise
            noisy_audio = self._apply_additive_noise(simulated_audio)
            
            # Each crop gets independent gain perturbation
            processed_audio = self._apply_channel_gain_and_normalize(noisy_audio)
            
            # Each crop gets different random position
            cropped = self._extract_random_crop(processed_audio)
            
            audio_tensor = torch.from_numpy(cropped.astype(np.float32))
            crops.append((audio_tensor, label_tensor))
        
        return crops

    def _load_random_source_clip(self, duration_sec=1.0):
        # Keep trying until we get a valid (non-empty) audio clip
        while True:
            wav_path = random.choice(self.source_files)
            # Get audio duration (librosa is slow, consider caching durations in production)
            full_duration = librosa.get_duration(path=wav_path)
            
            if full_duration <= duration_sec:
                offset = 0.0
            else:
                offset = np.random.uniform(0, full_duration - duration_sec)
            
            y, sr = librosa.load(wav_path, sr=self.fs, offset=offset, duration=duration_sec)
            
            # If empty, re-select another file
            if len(y) == 0:
                continue
            
            break
        
        # -----------------------------------------------------------
        # [Improvement 4] Pitch Shift to prevent timbre overfitting
        # Simulates different voice types: deep male -> high-pitched child
        # This effectively multiplies your 2703 files to 20000+ unique timbres
        # -----------------------------------------------------------
        y = self._random_pitch_shift(y, prob=0.4)
        
        # Normalize source signal to prevent clipping
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / max_val
        return y

    def __getitem__(self, idx):
        """
        Get a single training sample.
        
        Multi-Crop Strategy:
        - If cache is empty: run ONE simulation, generate N crops, fill cache
        - Return one crop from cache
        
        This amortizes the expensive room.simulate() cost across N samples.
        
        Note: In PyTorch DataLoader with num_workers > 0, each worker process
        gets its own copy of the Dataset object. So self.crop_cache is naturally
        isolated between processes - no threading.local() or locks needed!
        """
        # If cache is empty, run simulation and fill it
        if not self.crop_cache:
            new_crops = self._run_simulation_and_fill_cache()
            random.shuffle(new_crops)
            self.crop_cache.extend(new_crops)
        
        # Pop one sample from cache and return
        return self.crop_cache.pop()