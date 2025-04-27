from typing import List, Dict
import os

import librosa
import torch
from loguru import logger
from transformers import AutoTokenizer


from kimia_infer.models.tokenizer.whisper_Lv3.whisper import WhisperEncoder
from kimia_infer.models.tokenizer.glm4_tokenizer import Glm4Tokenizer
from kimia_infer.utils.data import KimiAContent
from kimia_infer.utils.special_tokens import instantiate_extra_tokens

class KimiAPromptManager:
    def __init__(self, model_path: str, kimia_token_offset: int):
        self.audio_tokenizer = Glm4Tokenizer("THUDM/glm-4-voice-tokenizer")
        self.audio_tokenizer = self.audio_tokenizer.to(torch.cuda.current_device())

        logger.info(f"Looking for resources in {model_path}")
        logger.info(f"Loading whisper model")

        self.whisper_model = WhisperEncoder(
            os.path.join(model_path, "whisper-large-v3"), mel_batch_size=20
        )
        self.whisper_model = self.whisper_model.to(torch.cuda.current_device())
        self.whisper_model = self.whisper_model.bfloat16()
        self.whisper_model.eval()

        logger.info(f"Loading text tokenizer")
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        self.extra_tokens = instantiate_extra_tokens(self.text_tokenizer)

        self.kimia_token_offset = kimia_token_offset

    def _tokenize_text(self, text):
        if text is None:
            return None
        token_ids = self.text_tokenizer.encode(text, bos=False, eos=False)
        return token_ids

    def _tokenize_audio(self, wav_path):
        wav_tokens = self.audio_tokenizer.tokenize(audio_path=wav_path)
        wav_tokens = wav_tokens + self.kimia_token_offset
        wav_tokens_list = wav_tokens.squeeze(0).cpu().numpy().tolist()
        return wav_tokens_list

    def extract_whisper_feat(self, wav: torch.Tensor | str):
        if isinstance(wav, str):
            wav = librosa.load(wav, sr=16000)[0]

            wav_tensor = torch.tensor(wav).unsqueeze(0)[:, :]
        elif isinstance(wav, torch.Tensor):
            wav_tensor = wav
        else:
            raise ValueError(f"Invalid wav type: {type(wav)}")
        assert self.whisper_model is not None
        wav_tensor = wav_tensor.to(torch.cuda.current_device())
        continous_feature = self.whisper_model.tokenize_waveform(wav_tensor)
        continous_feature = continous_feature.reshape(
            continous_feature.shape[0],
            int(continous_feature.shape[1] // 4),
            continous_feature.shape[2] * 4,
        )
        return continous_feature

    def tokenize_message(
        self,
        message,
        tokenize_role=True,
        has_ct_token=False,
        has_msg_end_token=False,
        extract_whisper_feature=False,
        output_type: str = "text",
    ):
        kimia_content_msg = KimiAContent()

        role = message["role"]

        if tokenize_role:
            if role == "user":
                kimia_content_msg.audio_append(self.extra_tokens.kimia_user_msg_start)
                kimia_content_msg.text_append(self.extra_tokens.kimia_text_blank)
            elif role == "assistant":
                kimia_content_msg.audio_append(
                    self.extra_tokens.kimia_assistant_msg_start
                )
                kimia_content_msg.text_append(self.extra_tokens.kimia_text_blank)
            else:
                raise NotImplementedError(f"role: {role}")

        if message["message_type"] == "text":
            text = message["content"]
            text_tokens = self._tokenize_text(text)

            kimia_content_msg.text_extend(text_tokens)
            kimia_content_msg.audio_extend(
                [self.extra_tokens.kimia_text_blank] * len(text_tokens)
            )

        elif message["message_type"] == "audio":
            audio_path = message["content"]
            speech_tokens = self._tokenize_audio(audio_path)

            kimia_content_msg.audio_append(self.extra_tokens.media_begin)
            kimia_content_msg.audio_extend(speech_tokens, is_continuous=True)
            kimia_content_msg.audio_append(self.extra_tokens.media_end)
            kimia_content_msg.text_extend(
                [self.extra_tokens.kimia_text_blank] * (len(speech_tokens) + 2)
            )

            if has_ct_token:
                if output_type == "text":
                    kimia_content_msg.audio_append(self.extra_tokens.kimia_speech_ct_id)
                else:
                    kimia_content_msg.audio_append(
                        self.extra_tokens.kimia_speech_ctd_id
                    )
                kimia_content_msg.text_append(self.extra_tokens.kimia_text_blank)

            if extract_whisper_feature:
                whisper_feature = self.extract_whisper_feat(audio_path)
                kimia_content_msg.continuous_feature.append(whisper_feature)
        elif message["message_type"] == None:
            pass
        else:
            raise NotImplementedError(f"message_type: {message['message_type']}")

        if has_msg_end_token:
            kimia_content_msg.audio_append(self.extra_tokens.msg_end)
            kimia_content_msg.text_append(self.extra_tokens.kimia_text_blank)

        assert (
            kimia_content_msg.is_valid()
        ), f"kimia_content_msg is not valid: {kimia_content_msg}"

        return kimia_content_msg

    def get_prompt(
        self, messages: List[Dict], output_type: str = "text"
    ) -> KimiAContent:
        """
        messages: List[Dict]
        messages[i] = {
            "role": "user" | "assistant" | "system",
            "content": str
        }
        """
        assert output_type in ["text", "both"]

        msgs: List[KimiAContent] = []
        tokenize_role = True
        has_ct_token = False
        has_msg_end_token = False

        previous_role = None
        for msg_idx, message in enumerate(messages):
            assert message["role"] in ["user", "assistant"]

            if previous_role is None:
                tokenize_role = True
            else:
                if message["role"] == previous_role:
                    tokenize_role = False
                else:
                    tokenize_role = True

            if msg_idx == len(messages) - 1:
                has_ct_token = True
                has_msg_end_token = True
            else:
                if messages[msg_idx + 1]["role"] != message["role"]:
                    has_ct_token = True
                    has_msg_end_token = True
                else:
                    has_ct_token = False
                    has_msg_end_token = False

            previous_role = message["role"]

            msg = self.tokenize_message(
                message=message,
                tokenize_role=tokenize_role,
                has_ct_token=has_ct_token,
                has_msg_end_token=has_msg_end_token,
                extract_whisper_feature=True,
                output_type=output_type,
            )
            msgs.append(msg)

        assistant_start_msg = self.tokenize_message(
            message={
                "role": "assistant",
                "message_type": None,
            },
            tokenize_role=True,
            has_ct_token=False,
            has_msg_end_token=False,
        )

        msgs.append(assistant_start_msg)

        ret_msg = msgs[0]

        for msg in msgs[1:]:
            ret_msg.merge(msg)

        return ret_msg
