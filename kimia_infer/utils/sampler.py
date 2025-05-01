from dataclasses import dataclass, field

import torch


@dataclass
class KimiASampler:

    audio_top_k: int = 5
    audio_temperature: float = 0.0
    audio_repetition_penalty: float = 1.0
    audio_repetition_window_size: int = 64
    text_top_k: int = 5
    text_temperature: float = 0.0
    text_repetition_penalty: float = 1.0
    text_repetition_window_size: int = 16

    def sample_audio_logits(
        self, logits: torch.Tensor, recent_tokens=None
    ) -> torch.Tensor:
        """Sample from audio logits with top-k, temperature and repetition penalty.

        Args:
            logits: Logits tensor of shape [batch_size, seq_len, vocab_size] or [batch_size, vocab_size]
            recent_tokens: Optional tensor of recent tokens for repetition penalty

        Returns:
            Sampled token ids
        """
        # Take the last token's logits if we have a sequence dimension
        if len(logits.shape) == 3:
            logits = logits[:, -1]

        # Apply repetition penalty if needed
        if (
            self.audio_repetition_penalty > 1.0
            and recent_tokens is not None
            and len(recent_tokens) > self.audio_repetition_window_size
        ):
            logits = logits[0]  # Assumes batch size of 1 for repetition penalty
            recent_window = recent_tokens[-self.audio_repetition_window_size :].long()

            # Gather scores of recent tokens
            scores = torch.gather(logits, dim=0, index=recent_window)

            # Apply penalty: if score < 0 multiply by penalty, otherwise divide by penalty
            scores = torch.where(
                scores < 0,
                scores * self.audio_repetition_penalty,
                scores / self.audio_repetition_penalty,
            )

            # Put the penalized scores back
            logits.scatter_(dim=0, index=recent_window, src=scores)
            logits = logits.unsqueeze(0)  # Add batch dimension back

        # Convert to probabilities with softmax
        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)

        # Apply temperature scaling if not greedy
        if self.audio_temperature > 1e-6:
            logprobs = logprobs / self.audio_temperature

            # Apply top-k sampling
            if self.audio_top_k > 0:
                # Get probabilities from logprobs
                probs = torch.exp(logprobs)

                # Select top-k probabilities and indices
                top_k_probs, top_k_indices = torch.topk(probs, self.audio_top_k, dim=-1)

                # Sample from the top-k distribution
                sampled_indices = torch.multinomial(top_k_probs, num_samples=1).squeeze(
                    1
                )
                next_token = top_k_indices.gather(
                    -1, sampled_indices.unsqueeze(-1)
                ).squeeze(-1)
            else:
                # Sample from the full distribution
                next_token = torch.multinomial(
                    torch.exp(logprobs), num_samples=1
                ).squeeze(1)
        else:
            # Greedy sampling (temperature = 0)
            next_token = torch.argmax(logprobs, dim=-1)

        return next_token

    def sample_text_logits(
        self, logits: torch.Tensor, recent_tokens=None
    ) -> torch.Tensor:
        """Sample from text logits with top-k, temperature and repetition penalty.

        Args:
            logits: Logits tensor of shape [batch_size, seq_len, vocab_size] or [batch_size, vocab_size]
            recent_tokens: Optional tensor of recent tokens for repetition penalty

        Returns:
            Sampled token ids
        """
        # Take the last token's logits if we have a sequence dimension
        if len(logits.shape) == 3:
            logits = logits[:, -1]

        # Apply repetition penalty if needed
        if (
            self.text_repetition_penalty > 1.0
            and recent_tokens is not None
            and len(recent_tokens) > self.text_repetition_window_size
        ):
            logits = logits[0]  # Assumes batch size of 1 for repetition penalty
            recent_window = recent_tokens[-self.text_repetition_window_size :].long()

            # Gather scores of recent tokens
            scores = torch.gather(logits, dim=0, index=recent_window)

            # Apply penalty: if score < 0 multiply by penalty, otherwise divide by penalty
            scores = torch.where(
                scores < 0,
                scores * self.text_repetition_penalty,
                scores / self.text_repetition_penalty,
            )

            # Put the penalized scores back
            logits.scatter_(dim=0, index=recent_window, src=scores)
            logits = logits.unsqueeze(0)  # Add batch dimension back

        # Convert to probabilities with softmax
        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)

        # Apply temperature scaling if not greedy
        if self.text_temperature > 1e-6:
            logprobs = logprobs / self.text_temperature

            # Apply top-k sampling
            if self.text_top_k > 0:
                # Get probabilities from logprobs
                probs = torch.exp(logprobs)

                # Select top-k probabilities and indices
                top_k_probs, top_k_indices = torch.topk(probs, self.text_top_k, dim=-1)

                # Sample from the top-k distribution
                sampled_indices = torch.multinomial(top_k_probs, num_samples=1).squeeze(
                    1
                )
                next_token = top_k_indices.gather(
                    -1, sampled_indices.unsqueeze(-1)
                ).squeeze(-1)
            else:
                # Sample from the full distribution
                next_token = torch.multinomial(
                    torch.exp(logprobs), num_samples=1
                ).squeeze(1)
        else:
            # Greedy sampling (temperature = 0)
            next_token = torch.argmax(logprobs, dim=-1)

        return next_token
