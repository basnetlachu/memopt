"""
Correctness validation tests for Stage 1

CRITICAL: Ensures Stage 1 optimizations produce IDENTICAL outputs to Stage 0.
No numerical differences are acceptable.
"""

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    # Mock pytest for standalone execution
    class pytest:
        class mark:
            @staticmethod
            def skipif(condition, reason=""):
                def decorator(func):
                    def wrapper(*args, **kwargs):
                        if condition:
                            print(f"⏭️  Skipped: {reason}")
                            return
                        return func(*args, **kwargs)
                    return wrapper
                return decorator

        @staticmethod
        def fail(message):
            raise AssertionError(message)


import torch
from memopt import OptimizedLLM


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
class TestOutputCorrectness:
    """Ensure Stage 0 and Stage 1 produce identical outputs"""

    def test_identical_outputs_simple_prompt(self):
        """Test that Stage 0 and Stage 1 produce identical outputs for simple prompt"""
        model_name = "gpt2"
        prompt = "The quick brown fox"
        max_tokens = 20

        model_stage0 = OptimizedLLM(
            model=model_name,
            optimization_level="conservative",
            enable_profiling=False
        )

        model_stage1 = OptimizedLLM(
            model=model_name,
            optimization_level="balanced",
            enable_profiling=False
        )

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        output_stage0 = model_stage0.generate(
            prompt,
            max_tokens=max_tokens,
            do_sample=False
        )

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        output_stage1 = model_stage1.generate(
            prompt,
            max_tokens=max_tokens,
            do_sample=False
        )

        assert output_stage0 == output_stage1, \
            f"Outputs differ!\nStage 0: {output_stage0}\nStage 1: {output_stage1}"

        print("✓ Outputs identical for simple prompt")

    def test_identical_outputs_multiple_prompts(self):
        """Test outputs are identical across multiple diverse prompts"""
        model_name = "gpt2"
        prompts = [
            "Once upon a time",
            "The capital of France is",
            "In mathematics, the number",
        ]
        max_tokens = 15

        model_stage0 = OptimizedLLM(
            model=model_name,
            optimization_level="conservative",
            enable_profiling=False
        )

        model_stage1 = OptimizedLLM(
            model=model_name,
            optimization_level="balanced",
            enable_profiling=False
        )

        for i, prompt in enumerate(prompts):
            torch.manual_seed(42 + i)
            torch.cuda.manual_seed(42 + i)

            output_stage0 = model_stage0.generate(
                prompt,
                max_tokens=max_tokens,
                do_sample=False
            )

            torch.manual_seed(42 + i)
            torch.cuda.manual_seed(42 + i)

            output_stage1 = model_stage1.generate(
                prompt,
                max_tokens=max_tokens,
                do_sample=False
            )

            assert output_stage0 == output_stage1, \
                f"Prompt {i} differs!\nStage 0: {output_stage0}\nStage 1: {output_stage1}"

        print(f"✓ Outputs identical for {len(prompts)} prompts")

    def test_no_numerical_drift(self):
        """Ensure no numerical drift over multiple generations"""
        model_name = "gpt2"
        prompt = "The answer is"
        max_tokens = 50

        model_stage0 = OptimizedLLM(
            model=model_name,
            optimization_level="conservative",
            enable_profiling=False
        )

        model_stage1 = OptimizedLLM(
            model=model_name,
            optimization_level="balanced",
            enable_profiling=False
        )

        for run in range(10):
            torch.manual_seed(100 + run)
            torch.cuda.manual_seed(100 + run)

            output_stage0 = model_stage0.generate(
                prompt,
                max_tokens=max_tokens,
                do_sample=False
            )

            torch.manual_seed(100 + run)
            torch.cuda.manual_seed(100 + run)

            output_stage1 = model_stage1.generate(
                prompt,
                max_tokens=max_tokens,
                do_sample=False
            )

            assert output_stage0 == output_stage1, \
                f"Run {run} differs! Numerical drift detected."

        print("✓ No numerical drift over 10 runs")


class TestMemoryCorrectness:
    """Ensure Stage 1 memory optimizations don't cause allocation failures"""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_no_out_of_memory_errors(self):
        """Ensure adaptive allocation doesn't cause OOM"""
        model_name = "gpt2"

        test_cases = [
            (1, 100),
            (4, 256),
            (8, 512),
        ]

        for num_prompts, max_tokens in test_cases:
            model = OptimizedLLM(
                model=model_name,
                optimization_level="balanced",
                expected_batch_size=num_prompts,
                expected_seq_len=max_tokens,
                enable_profiling=False
            )

            prompts = [f"Test prompt {i}" for i in range(num_prompts)]

            try:
                for prompt in prompts:
                    _ = model.generate(
                        prompt,
                        max_tokens=max_tokens,
                        do_sample=False
                    )

                print(f"✓ No OOM for {num_prompts} prompts × {max_tokens} tokens")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    pytest.fail(
                        f"OOM error with {num_prompts} prompts × {max_tokens} tokens"
                    )
                else:
                    raise


if __name__ == "__main__":
    print("=" * 70)
    print("CORRECTNESS VALIDATION TESTS")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("⚠️  WARNING: CUDA not available, tests will be skipped")
        exit(0)

    print("\n--- Testing Output Correctness ---")
    test_correctness = TestOutputCorrectness()
    test_correctness.test_identical_outputs_simple_prompt()
    test_correctness.test_identical_outputs_multiple_prompts()
    test_correctness.test_no_numerical_drift()

    print("\n--- Testing Memory Correctness ---")
    test_memory = TestMemoryCorrectness()
    test_memory.test_no_out_of_memory_errors()

    print("\n" + "=" * 70)
    print("✅ ALL CORRECTNESS TESTS PASSED")
    print("Stage 1 produces IDENTICAL outputs to Stage 0")
    print("=" * 70)
