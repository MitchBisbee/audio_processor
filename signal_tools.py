import logging
import numpy as np
import scipy.constants
import scipy.io.wavfile
import scipy.signal
import matplotlib.pyplot as plt
import sounddevice as sd

# need to adjust some things so that the dtft of music files can be plotted
# then this should be ready to be used as a backend for the  audio app
logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class AudioAnalyzer:
    """## A class for analyzing audio files.
    """

    def __init__(self, filename):
        try:
            self.sr, self.y = scipy.io.wavfile.read(filename)
            self.filtered_signal = None
            self.filename = filename
        except Exception as e:
            logging.error("Failed to load audio file %s: %s", filename, e)
            raise IOError(f'Could not read file {filename}: {e}') from e

    def apply_bandpass_filter(self, lowcut: int, highcut: int, order: int) -> None:
        """## Creates and apply's a bad pass filter to an audio file.

        ### Args:
            - `lowcut (int)`: low cutoff frequency
            - `highcut (int)`: high cutoff frequency 
            - `order (int)`: filter order
        """
        try:
            sos = scipy.signal.butter(
                order, [lowcut, highcut], btype='band', fs=self.sr, output='sos')
            self.filtered_signal = scipy.signal.sosfilt(sos, self.y)
        except ValueError as e:
            logging.error("Error applying bandpass filter: %s", e)
            raise

    def apply_highpass_filter(self, order: int, cutoff: int) -> None:
        """## Creates and apply's a high pass filter to an audio file.

        ### Args:
            - `order (int)`: filter order
            - `cutoff (int)`: cutoff frequency
        """
        try:
            w_n = cutoff / (self.sr/2)
            sos = scipy.signal.butter(
                N=order, Wn=w_n, btype='high', fs=self.sr, output='sos')
            self.filtered_signal = scipy.signal.sosfilt(sos, self.y)
        except ValueError as e:
            logging.error("Error applying highpass filter: %s", e)
            raise

    def apply_lowpass_filter(self, cutoff: int, order: int) -> None:
        """## Creates and apply's a lowpass filter to an audio file.

        ### Args:
            - `cutoff (int)`: cutoff frequency
            - `order (int)`: filter order
        """
        try:
            w_n = cutoff / (self.sr/2)
            sos = scipy.signal.butter(
                order, Wn=w_n, btype='low', fs=self.sr, output='sos')
            self.filtered_signal = scipy.signal.sosfilt(sos, self.y)
        except ValueError as e:
            logging.error("Error applying lowpass filter: %s", e)
            raise

    def apply_bandstop_filter(self, cutoff: tuple, order: int) -> None:
        """## Creates and apply's a band stop filter to an audio file.

        ### Args:
            - `cutoff (tuple)`: (low cutoff frequency , high cutoff frequency)
            - `order (int)`: filter order

        ### Raises:
            - `ValueError`: Cutoff must be a tuple (low_freq, high_freq)
        """
        try:
            if not isinstance(cutoff, tuple) or len(cutoff) != 2:
                raise ValueError(
                    "Cutoff must be a tuple (low_freq, high_freq)")
            sos = scipy.signal.butter(
                order, cutoff, btype='bandstop', fs=self.sr, output='sos')
            self.filtered_signal = scipy.signal.sosfilt(sos, self.y)
        except ValueError as e:
            logging.error("Error applying bandstop filter: %s", e)
            raise

    def apply_fourier_transform(self) -> np.NDArray[np.complex128]:
        """## Compute the one-dimensional discrete Fourier Transform.

            This function computes the one-dimensional n-point discrete Fourier 
            Transform (DFT) with the efficient Fast Fourier Transform (FFT) 
            algorithm [CT].

        ### Returns:
            - `np.NDArray[np.complex128]`:  complex ndarray
        """
        try:
            fft = np.fft.fft(self.y)
            return fft
        except Exception as e:
            logging.error("Error computing Fourier Transform: %s", e)
            raise

    def display_filter_frequency_response(self, filter_type: str, order: int,
                                          cutoff: int | tuple) -> None:
        """## Plots the frequency response of the input filter type.

        ### Args:
            - `filter_type (str)`: 'high', 'low', 'band', or 'bandstop'
            - `order (int)`: filter order
            - `cutoff (int | tuple)`: cut off frequency for your filter

        ### Raises:
            - `ValueError`: "Cutoff for 'band' type must be a tuple (low, high)"
            - `ValueError`:  "Cutoff for 'bandstop' type must be a tuple (low, high)"
            - `ValueError`:  "filter_type must be 'high', 'low', 'band', or 'bandstop'"
        """
        w_n = cutoff / (self.sr / 2)
        if filter_type.lower() == "high":
            b, a = scipy.signal.butter(
                order, Wn=w_n, btype="high", output='ba')
        elif filter_type.lower() == "low":
            b, a = scipy.signal.butter(order, Wn=w_n, btype="low", output='ba')
        elif filter_type.lower() == "band":
            if not isinstance(cutoff, tuple):
                raise ValueError(
                    "Cutoff for 'band' type must be a tuple (low, high)")
            b, a = scipy.signal.butter(
                order, Wn=cutoff, btype="band", fs=self.sr, output='ba')
        elif filter_type.lower() == "bandstop":
            if not isinstance(cutoff, tuple):
                raise ValueError(
                    "Cutoff for 'bandstop' type must be a tuple (low, high)")
            b, a = scipy.signal.butter(
                order, Wn=cutoff, btype="bandstop", fs=self.sr, output='ba')
        else:
            raise ValueError(
                "filter_type must be 'high', 'low', 'band', or 'bandstop'")
        w, h = scipy.signal.freqz(b, a)
        self._plot(f"{filter_type} pass filter response",
                   x=w * self.sr / (2 * np.pi), y=np.abs(h),
                   x_label="Frequency (Hz)", y_label="Magnitude")

    def display_filter_impulse_response(self, filter_type: str, order: int,
                                        cutoff: int | tuple) -> None:
        """## Display the impulse response of your filter design.

        ### Args:
            - `filter_type (str)`: 'high', 'low', 'band', or 'bandstop'
            - `order (int)`: order of you filter
            - `cutoff (int | tuple)`: cutoff frequency of your filter

        ### Raises:
            - `ValueError`: "Cutoff for 'band' type must be a tuple (low, high)"
            - `ValueError`: "Cutoff for 'bandstop' type must be a tuple (low, high)"
            - `ValueError`:  "filter_type must be 'high', 'low', 'band', or 'bandstop'"
        """

        try:
            # Calculate normalized cutoff frequency or frequencies
            w_n = cutoff / (self.sr / 2) if isinstance(cutoff,
                                                       (int, float)) else np.array(cutoff) / (self.sr / 2)
            # Design the filter based on type and cutoff
            if filter_type.lower() == "high":
                b, a = scipy.signal.butter(
                    order, Wn=w_n, btype="high", output='ba')
            elif filter_type.lower() == "low":
                b, a = scipy.signal.butter(
                    order, Wn=w_n, btype="low", output='ba')
            elif filter_type.lower() == "band":
                if not isinstance(cutoff, tuple):
                    raise ValueError(
                        "Cutoff for 'band' type must be a tuple (low, high)")
                b, a = scipy.signal.butter(
                    order, Wn=w_n, btype="band", fs=self.sr, output='ba')
            elif filter_type.lower() == "bandstop":
                if not isinstance(cutoff, tuple):
                    raise ValueError(
                        "Cutoff for 'bandstop' type must be a tuple (low, high)")
                b, a = scipy.signal.butter(
                    order, Wn=w_n, btype="bandstop", fs=self.sr, output='ba')
            else:
                raise ValueError(
                    "filter_type must be 'high', 'low', 'band', or 'bandstop'")

            # Generate an impulse signal with the same length as the audio data
            impulse = np.zeros(len(self.y))
            impulse[0] = 1  # Creating an impulse at the start
            # Compute the filter's response to the impulse
            response = scipy.signal.lfilter(b, a, impulse)

            # Calculate the time array corresponding to each sample in the response
            time_array = np.arange(len(response)) / self.sr

            # Plot the response
            self._plot(f"Order: {order} {filter_type.capitalize()} Pass Filter - Impulse Response",
                       x=time_array, y=response, x_label="Time (seconds)", y_label="Amplitude")
        except ValueError as e:
            logging.error("Invalid filter parameters: %s", e)
            raise

    def display_filtered_audio(self, filter_type: str, order: int, cutoff: int) -> None:
        """## Displays the time domain plot of the your filtered audio file.

        ### Args:
            - `filter_type (str)`: high', 'low', 'band', or 'bandstop
            - `order (int)`: order of you filter
            - `cutoff (int | tuple)`: cutoff frequency of your filter

        ### Raises:
            - `ValueError`: "Cutoff for 'band' type must be a tuple (low, high)"
            - `ValueError`: "Cutoff for 'bandstop' type must be a tuple (low, high)"
            - `ValueError`: "filter_type must be 'high', 'low', 'band', or 'bandstop'"
        """
        try:
            # Calculate normalized cutoff frequency or frequencies
            w_n = cutoff / (self.sr / 2) if isinstance(cutoff,
                                                       (int, float)) else np.array(cutoff) / (self.sr / 2)
            # Design the filter based on type and cutoff
            if filter_type.lower() == "high":
                b, a = scipy.signal.butter(
                    order, Wn=w_n, btype="high", output='ba')
            elif filter_type.lower() == "low":
                b, a = scipy.signal.butter(
                    order, Wn=w_n, btype="low", output='ba')
            elif filter_type.lower() == "band":
                if not isinstance(cutoff, tuple):
                    raise ValueError(
                        "Cutoff for 'band' type must be a tuple (low, high)")
                b, a = scipy.signal.butter(
                    order, Wn=w_n, btype="band", fs=self.sr, output='ba')
            elif filter_type.lower() == "bandstop":
                if not isinstance(cutoff, tuple):
                    raise ValueError(
                        "Cutoff for 'bandstop' type must be a tuple (low, high)")
                b, a = scipy.signal.butter(
                    order, Wn=w_n, btype="bandstop", fs=self.sr, output='ba')
            else:
                raise ValueError(
                    "filter_type must be 'high', 'low', 'band', or 'bandstop'")

            filt_speech = scipy.signal.lfilter(b, a, self.y)
            # Calculate the time array corresponding to each sample in the response
            time_array = np.arange(len(self.y)) / self.sr

            # Plot the response
            self._plot(f"Order: {order} {filter_type.capitalize()} Pass Filter - Time Domain Response",
                       x=time_array, y=filt_speech, x_label="Time (seconds)", y_label="Amplitude")
        except ValueError as e:
            logging.error("Invalid filter parameters: %s", e)
            raise

    def display_spectral_content(self) -> None:
        """## Plots and displays the specteral content of your audio file.
        """
        # Display the spectral content of the audio
        f, t, Sxx = scipy.signal.spectrogram(self.y, self.sr)
        plt.pcolormesh(t, f, 10 * np.log10(Sxx))
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title('Spectral Content of ' + self.filename)
        plt.colorbar(label='Intensity [dB]')
        plt.show()

    def display_norm_wave_content(self) -> None:
        """## Plots and displays the normalized waveform of your audio file.
        """
        plt.plot(np.arange(len(self.y)) / self.sr,
                 self.y / np.max(np.abs(self.y)))
        plt.title('Normalized Waveform of ' + self.filename)
        plt.xlabel('Time [seconds]')
        plt.ylabel('Amplitude [normalized]')
        plt.show()

    def _plot(self, title: str, x: int | list, y: int | list, x_label: str,
              y_label: str) -> None:
        """## Private method used to plot filter responses.

        ### Args:
            - `title (str)`: plot title
            - `x (int | list)`: x-axis data
            - `y (int | list)`: y-axis data
            - `x_label (str)`: x-axis label
            - `y_label (str)`: y-axis label
        """
        plt.plot(x, y)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid()
        plt.show()

    def play_audio(self, filtered_signal=False) -> None:
        """## Plays your regular audio. If filtered_signal = True the
              filtered of the file is played based off your filter design.

        ### Args:
            - `filtered_signal (bool, optional)`: Plays the filtered signal 
               from your filter designs. Defaults to False.

        ### Raises:
            - `ValueError`: _description_
        """
        if filtered_signal:
            if self.filtered_signal is not None and filtered_signal.size > 0:
                sd.play(data=filtered_signal, samplerate=self.sr)
            else:
                raise ValueError(
                    "A filter must be applied before it can be played.")
        else:
            sd.play(data=self.y, samplerate=self.sr)

    def display_dtft_magnitude(self) -> None:
        """## Plots the dtft magnitude of the file
        """

        b, a = scipy.signal.freqz(self.y, 1, len(self.y), fs=self.sr)
        self._plot(f"DTFT Mag: {self.filename}", x=b, y=abs(
            a), x_label="Frequency (Hz)", y_label="Magnitude")

    def save_audio_file(self, use_filtered=True, output_filename=None) -> None:
        """## Saves the filtered version of your file or the normal version.

        ### Args:
            - `use_filtered (bool, optional)`: Saves the filtered version of 
                the file. Defaults to True.
            - `output_filename (str, optional)`: _description_. Defaults to None.

        ### Raises:
            - `IOError`: _description_
        """
        try:
            data_to_save = self.filtered_signal if use_filtered and self.filtered_signal is not None else self.y
            if output_filename is None:
                suffix = '_filtered' if use_filtered and self.filtered_signal is not None else '_original'
                output_filename = self.filename.replace(
                    '.wav', f'{suffix}.wav')
            scipy.io.wavfile.write(output_filename, self.sr, data_to_save)
            print(f"File saved as: {output_filename}")
        except Exception as e:
            logging.error("Error saving audio file: %s", e)
            raise IOError(f'Failed to save file: {e}') from e


if __name__ == "__main__":
    test = AudioAnalyzer("speechlab3.wav")
    test.play_audio()
    test.display_dtft_magnitude()
    test.display_norm_wave_content()
    test.display_spectral_content()
    test.display_filter_frequency_response('high', 2, 500)
    test.display_filtered_audio("high", 24, 500)
