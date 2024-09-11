from human_eval.data import write_jsonl
import json


num_samples_per_task = 1
with open('chatCodeQwenPython.jsonl', 'r') as file:
    for genPython in file:
        genPython = json.loads(genPython)
        generation = genPython['generation']
        if generation is not None:
            if generation.find("```python") != -1:
                generation = generation[generation.find("```python")+len("```python")+len("\n"):generation.find("```",generation.find("```python")+len("```python")+len("\n"))]
            ans = ""
            if generation.startswith('\n'):
                generation = generation[len('\n'):]
            for k in generation.split('\n'):
                if k.startswith('```'):
                    continue
                if k.startswith("def"):
                    continue
                if k.startswith("from"):
                    continue
                if k.startswith("import"):
                    continue
                ans = ans + k + '\n'
            generation = ans

        for _ in range(num_samples_per_task):
            samples = [
                dict(
                    task_id=genPython['task_id'],
                    generation=generation,
                    canonical_solution=genPython["canonical_solution"],
                    declaration=genPython["declaration"],
                    example_test=genPython["example_test"],
                    prompt=genPython["prompt"],
                    test=genPython["test"],
                    text=genPython["text"]
                )
            ]
            write_jsonl("test2ChatCodeQwenPython.jsonl", samples, True)