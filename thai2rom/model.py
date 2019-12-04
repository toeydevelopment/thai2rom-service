import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, load_model
from pythainlp.transliterate import romanize

class ThaiTransliterator:
    def __init__(self):
        """
        Transliteration of Thai words
        Now supports Thai to Latin (romanization)
        """
        self.input_token_index = {
            ' ': 0, '!': 1, '"': 2, '(': 3, ')': 4,
            '-': 5, '.': 6, '0': 7, '1': 8, '2': 9,
            '3': 10, '4': 11, '5': 12, '6': 13, '7': 14,
            '8': 15, '9': 16, '\xa0': 17, 'ก': 18, 'ข': 19,
            'ฃ': 20, 'ค': 21, 'ฅ': 22, 'ฆ': 23, 'ง': 24,
            'จ': 25, 'ฉ': 26, 'ช': 27, 'ซ': 28, 'ฌ': 29,
            'ญ': 30, 'ฎ': 31, 'ฏ': 32, 'ฐ': 33, 'ฑ': 34,
            'ฒ': 35, 'ณ': 36, 'ด': 37, 'ต': 38, 'ถ': 39,
            'ท': 40, 'ธ': 41, 'น': 42, 'บ': 43, 'ป': 44,
            'ผ': 45, 'ฝ': 46, 'พ': 47, 'ฟ': 48, 'ภ': 49,
            'ม': 50, 'ย': 51, 'ร': 52, 'ฤ': 53, 'ล': 54,
            'ฦ': 55, 'ว': 56, 'ศ': 57, 'ษ': 58, 'ส': 59,
            'ห': 60, 'ฬ': 61, 'อ': 62, 'ฮ': 63, 'ฯ': 64,
            'ะ': 65, 'ั': 66, 'า': 67, 'ำ': 68, 'ิ': 69,
            'ี': 70, 'ึ': 71, 'ื': 72, 'ุ': 73, 'ู': 74,
            'ฺ': 75, 'เ': 76, 'แ': 77, 'โ': 78, 'ใ': 79,
            'ไ': 80, 'ๅ': 81, 'ๆ': 82, '็': 83, '่': 84,
            '้': 85, '๊': 86, '๋': 87, '์': 88, 'ํ': 89, '๙': 90
        }
        self.target_token_index = {
            '\t': 0, '\n': 1, ' ': 2, '!': 3, '"': 4,
            '(': 5, ')': 6, '-': 7, '0': 8, '1': 9,
            '2': 10, '3': 11, '4': 12, '5': 13,
            '6': 14, '7': 15, '8': 16, '9': 17, 'a': 18,
            'b': 19, 'c': 20, 'd': 21, 'e': 22, 'f': 23,
            'g': 24, 'h': 25, 'i': 26, 'k': 27, 'l': 28,
            'm': 29, 'n': 30, 'o': 31, 'p': 32, 'r': 33,
            's': 34, 't': 35, 'u': 36, 'w': 37, 'y': 38
        }
        self.reverse_input_char_index = dict(
            (i, char) for char, i in self.input_token_index.items()
        )
        self.reverse_target_char_index = dict(
            (i, char) for char, i in self.target_token_index.items()
        )
        self.batch_size = 64
        self.epochs = 100
        self.latent_dim = 256
        self.num_encoder_tokens = 91
        self.num_decoder_tokens = 39
        self.max_encoder_seq_length = 20
        self.max_decoder_seq_length = 22

        # Restore the model and construct the encoder and decoder.
        self.model = load_model("model/thai2rom-v2.hdf5")
        self.encoder_inputs = self.model.input[0]  # input_1
        self.encoder_outputs, self.state_h_enc, self.state_c_enc = self.model.layers[
            2
        ].output  # lstm_1
        self.encoder_states = [self.state_h_enc, self.state_c_enc]
        self.encoder_model = Model(
            self.encoder_inputs, self.encoder_states
        )
        self.decoder_inputs = self.model.input[1]  # input_2
        self.decoder_state_input_h = Input(
            shape=(self.latent_dim,), name="input_3"
        )
        self.decoder_state_input_c = Input(
            shape=(self.latent_dim,), name="input_4"
        )
        self.decoder_states_inputs = [
            self.decoder_state_input_h,
            self.decoder_state_input_c,
        ]
        self.decoder_lstm = self.model.layers[3]
        self.decoder_outputs, self.state_h_dec, self.state_c_dec = self.decoder_lstm(
            self.decoder_inputs, initial_state=self.decoder_states_inputs
        )
        self.decoder_states = [self.state_h_dec, self.state_c_dec]
        self.decoder_dense = self.model.layers[4]
        self.decoder_outputs = self.decoder_dense(self.decoder_outputs)
        self.decoder_model = Model(
            [self.decoder_inputs] + self.decoder_states_inputs,
            [self.decoder_outputs] + self.decoder_states,
        )

    def decode_sequence(self, input_seq):
        states_value = self.encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        target_seq[0, 0, self.target_token_index["\t"]] = 1.
        stop_condition = False
        decoded_sentence = ""

        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + states_value
            )
            sampled_token_index = np.argmax(
                output_tokens[0, -1, :]
            )
            sampled_char = self.reverse_target_char_index[
                sampled_token_index
            ]
            decoded_sentence += sampled_char
            if (
                sampled_char == "\n"
                or len(decoded_sentence) > self.max_decoder_seq_length
            ):
                stop_condition = True
            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.
            states_value = [h, c]
        return decoded_sentence

    def encode_input(self, name):
        test_input = np.zeros(
            (1, self.max_encoder_seq_length, self.num_encoder_tokens),
            dtype="float32",
        )
        for t, char in enumerate(name):
            test_input[0, t, self.input_token_index[char]] = 1.
        return test_input

    def romanize(self, text):
        """
        :param str text: Thai text to be romanized
        :return: English (more or less) text that spells out how the Thai text should be pronounced.
        """
        return self.decode_sequence(self.encode_input(text)).strip()