# test_sim_examples.py

from sim import SIM


def test_sim_with_examples():
    """Test SIM with multiple realistic examples."""

    sim = SIM(model="gemma3:1b")

    test_cases = [
        {
            "prompt": "Berlin is the capital of [RETRIEVE: capital of Germany]. It has a rich",
            "expected_entity": "Germany",
            "description": "Geography - Capital retrieval",
        },
        {
            "prompt": "The French Revolution occurred in [RETRIEVE: French Revolution year]. This event",
            "expected_entity": "1789",
            "description": "History - Date retrieval",
        },
        {
            "prompt": "The Eiffel Tower is located in [RETRIEVE: Eiffel Tower location]. It is",
            "expected_entity": "Paris",
            "description": "Culture - Landmark location",
        },
        {
            "prompt": "France won the FIFA World Cup in [RETRIEVE: France World Cup victory]. The team",
            "expected_entity": "1998",
            "description": "Sports - Championship year",
        },
    ]

    print("=" * 70)
    print("SIM Multi-Example Test Suite")
    print("=" * 70)

    for i, test in enumerate(test_cases, 1):
        print(f"\n[Test {i}] {test['description']}")
        print(f"Prompt: {test['prompt'][:60]}...")
        print("-" * 70)

        sim.run(test["prompt"])

        print("\n" + "=" * 70)


if __name__ == "__main__":
    test_sim_with_examples()
