"""Evaluation framework: 20 questions requiring specific podcast knowledge."""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import ollama
from rich.console import Console
from rich.table import Table

from podcast_rag.rag import query as rag_query

EVAL_PATH = Path(__file__).resolve().parents[2] / "data" / "eval_questions.json"
RESULTS_PATH = Path(__file__).resolve().parents[2] / "data" / "eval_results.json"

EVAL_QUESTIONS = [
    {
        "id": 1,
        "question": "What did Andrew NG say about the biggest bottlenecks in AI on the Twenty Minute VC?",
        "ground_truth": "Andrew NG discussed AI's rapid evolution, productivity loss from context switching, and the need to shift from consuming to building AI.",
        "category": "specific_guest",
    },
    {
        "id": 2,
        "question": "What is Dreamer and what approach does it take to agent development according to Latent Space?",
        "ground_truth": "Dreamer is a Personal Agent OS that takes a 'taste'-driven design approach where human creativity and intuition guide agent-based software development, with multi-agent workflow collaboration.",
        "category": "specific_product",
    },
    {
        "id": 3,
        "question": "What did Terence Tao discuss about AI's impact on mathematics on the Dwarkesh Podcast?",
        "ground_truth": "Terence Tao explored how AI is reshaping mathematics, shifting mathematicians from routine computation to higher-level problem-solving and the adaptability required for mathematical progress.",
        "category": "specific_guest",
    },
    {
        "id": 4,
        "question": "What were the key announcements from Amazon Reinvent 2024 regarding AI, according to The AI Breakdown?",
        "ground_truth": "Amazon launched AI Agents, announced Trainium 3 Ultra and Trainium 4 chips, and introduced AI Factories Service for specialized AI computing.",
        "category": "specific_event",
    },
    {
        "id": 5,
        "question": "What is SambaNova's approach to AI computing infrastructure as discussed on Tech Disruptors?",
        "ground_truth": "SambaNova emphasizes energy efficiency and data sovereignty in AI computing, focusing on foundation models with a privacy-first approach.",
        "category": "specific_company",
    },
    {
        "id": 6,
        "question": "What did Grady Booch say about the 'third golden age of software engineering' on The Pragmatic Engineer?",
        "ground_truth": "Grady Booch explored AI anxieties through the lens of historical technological shifts, discussing systems theory, agent programming, and how this era parallels previous transformations.",
        "category": "specific_guest",
    },
    {
        "id": 7,
        "question": "What did Alex Blania discuss about 'proof of human' on the a16z Podcast?",
        "ground_truth": "Alex Blania discussed verifying users are human beings, the challenges from AI bots, and implications for social media and democracy through World's identity network.",
        "category": "specific_guest",
    },
    {
        "id": 8,
        "question": "What were the findings from Lenny's large-scale AI productivity survey?",
        "ground_truth": "AI tools are significantly boosting workplace productivity across roles (PMs, engineers, designers, founders), with widespread adoption and strong market fit for AI tools.",
        "category": "specific_research",
    },
    {
        "id": 9,
        "question": "What did Tom Tunguz from Theory Ventures say about the state of data and AI on Dataframed?",
        "ground_truth": "Tom Tunguz discussed conversational AI, AI-specific programming languages, and automated data engineering as key trends in the data and AI landscape.",
        "category": "specific_guest",
    },
    {
        "id": 10,
        "question": "What is Kumo's approach to time series forecasting as covered on The Data Exchange?",
        "ground_truth": "Kumo uses transformer models trained on relational data for time series forecasting, with a focus on synthetic data generation and relational time series patterns.",
        "category": "specific_company",
    },
    {
        "id": 11,
        "question": "What did Dan Sundheim discuss about investment philosophy on Invest Like the Best?",
        "ground_truth": "Dan Sundheim emphasized a 5-10 year investment horizon, the importance of leadership qualities, and geopolitical risk assessment particularly around Taiwan/China.",
        "category": "specific_guest",
    },
    {
        "id": 12,
        "question": "What is Decagon and how does it approach LLMs for enterprises, according to Invest Like the Best?",
        "ground_truth": "Decagon builds specialized AI infrastructure for enterprises using LLMs, focusing on data governance, security, and enterprise-grade deployment of large language models.",
        "category": "specific_company",
    },
    {
        "id": 13,
        "question": "What did Werner Vogels predict about technology trends for 2026 on the AWS Podcast?",
        "ground_truth": "Werner Vogels emphasized human-centered innovation, the importance of diversity and education, and cybersecurity challenges from quantum computing.",
        "category": "specific_guest",
    },
    {
        "id": 14,
        "question": "What is Bauplan's 'Git for Data' concept as discussed on The Joe Reis Show?",
        "ground_truth": "Bauplan proposes a new lakehouse architecture for AI agents, applying Git-like version control concepts to data management.",
        "category": "specific_company",
    },
    {
        "id": 15,
        "question": "What did Tim Ferriss say about why he walked away from angel investing on the Twenty Minute VC?",
        "ground_truth": "Tim Ferriss discussed holistic success beyond financial returns, the difference between emulation and understanding, and burnout and mental health in the context of walking away from angel investing after Uber.",
        "category": "specific_guest",
    },
    {
        "id": 16,
        "question": "What were the key topics in the Hard Fork episode about the 'Great AI Build-Out'?",
        "ground_truth": "The episode covered the massive AI infrastructure build-out, H-1B visa debates, and TikTok 'rapture talk' phenomenon fueling social media misinformation and collective anxiety.",
        "category": "specific_episode",
    },
    {
        "id": 17,
        "question": "How is Harvey scaling AI for legal services according to the No Priors podcast?",
        "ground_truth": "Harvey is developing AI-powered tools for legal services, focusing on scaling AI solutions for next-generation law firms with rapid growth.",
        "category": "specific_company",
    },
    {
        "id": 18,
        "question": "What did the Fray paper discuss about concurrency testing on the Disseminate podcast?",
        "ground_truth": "Fray is an efficient general-purpose concurrency testing tool for JVM, addressing the disconnect between academic concurrency testing research and practical software development needs.",
        "category": "specific_research",
    },
    {
        "id": 19,
        "question": "What is the Netflix/Warner Bros. Discovery acquisition analysis from Big Technology Podcast?",
        "ground_truth": "The podcast critically examined the Netflix/WBD acquisition trend alongside prediction markets and data-driven decision making in the tech industry.",
        "category": "specific_event",
    },
    {
        "id": 20,
        "question": "What did Adobe's CFO reveal about the company's AI strategy on Tech Disruptors?",
        "ground_truth": "Adobe's CFO detailed the company's strong financial health, share repurchase program, and growth strategy centered on AI accelerating Creative Cloud.",
        "category": "specific_company",
    },
]

SCORING_PROMPT = """Score this answer against the ground truth on a 0-2 scale:
- 0: Wrong or completely irrelevant
- 1: Partially correct, some relevant information but missing key details
- 2: Fully correct, captures the key facts from ground truth

Ground truth: {ground_truth}

Answer to score: {answer}

Respond with ONLY a single number: 0, 1, or 2."""


@dataclass
class EvalResult:
    question_id: int
    question: str
    category: str
    ground_truth: str
    rag_answer: str
    rag_score: int
    rag_time: float
    baseline_answer: str
    baseline_score: int


def score_answer(answer: str, ground_truth: str) -> int:
    """Use LLM to score an answer against ground truth."""
    response = ollama.chat(
        model="gemma3",
        messages=[{
            "role": "user",
            "content": SCORING_PROMPT.format(ground_truth=ground_truth, answer=answer),
        }],
    )
    text = response.message.content.strip()
    for char in text:
        if char in "012":
            return int(char)
    return 0


def get_baseline_answer(question: str) -> str:
    """Get vanilla LLM answer without RAG context."""
    response = ollama.chat(
        model="gemma3",
        messages=[{
            "role": "user",
            "content": question,
        }],
    )
    return response.message.content


def run_eval(save: bool = True) -> list[EvalResult]:
    """Run full evaluation: RAG vs baseline on all 20 questions."""
    console = Console()
    results: list[EvalResult] = []

    console.print(f"\n[bold]Running evaluation on {len(EVAL_QUESTIONS)} questions...[/bold]\n")

    for q in EVAL_QUESTIONS:
        console.print(f"  Q{q['id']}: {q['question'][:70]}...")

        # RAG answer
        start = time.time()
        rag_result = rag_query(q["question"], k=5, use_expansion=False, use_rerank=False)
        rag_time = time.time() - start
        rag_score = score_answer(rag_result.answer, q["ground_truth"])

        # Baseline answer
        baseline_answer = get_baseline_answer(q["question"])
        baseline_score = score_answer(baseline_answer, q["ground_truth"])

        result = EvalResult(
            question_id=q["id"],
            question=q["question"],
            category=q["category"],
            ground_truth=q["ground_truth"],
            rag_answer=rag_result.answer,
            rag_score=rag_score,
            rag_time=rag_time,
            baseline_answer=baseline_answer,
            baseline_score=baseline_score,
        )
        results.append(result)

        marker_rag = "✅" if rag_score == 2 else "⚠️" if rag_score == 1 else "❌"
        marker_base = "✅" if baseline_score == 2 else "⚠️" if baseline_score == 1 else "❌"
        console.print(f"    RAG: {marker_rag} ({rag_score}/2)  Baseline: {marker_base} ({baseline_score}/2)  [{rag_time:.1f}s]")

    # Summary table
    console.print()
    table = Table(title="Evaluation Results")
    table.add_column("Q#", style="bold", width=4)
    table.add_column("Category", style="cyan", width=18)
    table.add_column("RAG", justify="center", width=5)
    table.add_column("Baseline", justify="center", width=8)
    table.add_column("Time", justify="right", width=6)

    for r in results:
        rag_style = "green" if r.rag_score == 2 else "yellow" if r.rag_score == 1 else "red"
        base_style = "green" if r.baseline_score == 2 else "yellow" if r.baseline_score == 1 else "red"
        table.add_row(
            str(r.question_id),
            r.category,
            f"[{rag_style}]{r.rag_score}[/{rag_style}]",
            f"[{base_style}]{r.baseline_score}[/{base_style}]",
            f"{r.rag_time:.1f}s",
        )

    console.print(table)

    # Totals
    rag_total = sum(r.rag_score for r in results)
    baseline_total = sum(r.baseline_score for r in results)
    max_score = len(results) * 2
    console.print(f"\n[bold]RAG Score: {rag_total}/{max_score} ({rag_total/max_score*100:.0f}%)[/bold]")
    console.print(f"[bold]Baseline Score: {baseline_total}/{max_score} ({baseline_total/max_score*100:.0f}%)[/bold]")
    console.print(f"[bold]Advantage: +{rag_total - baseline_total} points[/bold]")

    # Category breakdown
    categories: dict[str, dict[str, list[int]]] = {}
    for r in results:
        if r.category not in categories:
            categories[r.category] = {"rag": [], "baseline": []}
        categories[r.category]["rag"].append(r.rag_score)
        categories[r.category]["baseline"].append(r.baseline_score)

    console.print("\n[bold]By Category:[/bold]")
    for cat, scores in sorted(categories.items()):
        rag_avg = sum(scores["rag"]) / len(scores["rag"])
        base_avg = sum(scores["baseline"]) / len(scores["baseline"])
        console.print(f"  {cat}: RAG {rag_avg:.1f} vs Baseline {base_avg:.1f}")

    if save:
        serialized = [
            {
                "question_id": r.question_id,
                "question": r.question,
                "category": r.category,
                "ground_truth": r.ground_truth,
                "rag_answer": r.rag_answer,
                "rag_score": r.rag_score,
                "rag_time": r.rag_time,
                "baseline_answer": r.baseline_answer,
                "baseline_score": r.baseline_score,
            }
            for r in results
        ]
        RESULTS_PATH.write_text(json.dumps(serialized, indent=2, ensure_ascii=False))
        console.print(f"\n[dim]Results saved to {RESULTS_PATH}[/dim]")

    return results


if __name__ == "__main__":
    run_eval()
