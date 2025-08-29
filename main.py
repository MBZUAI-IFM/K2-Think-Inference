from openai import AsyncOpenAI
from pydantic import BaseModel
import asyncio
import uuid
import json
from openai.types.chat.chat_completion import ChatCompletion
from typing import List, Callable, Awaitable, Any
import time
import logging


class Env:

    SOLVER_LLM_API_KEY: str = ''
    SOLVER_LLM_BASE_URL: str = ''
    SOLVER_LLM_MODEL: str = "K2-Think"

    PLANNER_LLM_API_KEY: str = ''
    PLANNER_LLM_BASE_URL: str = ''
    PLANNER_LLM_MODEL: str = ''

    SOLVER_PROMPT: str = "You are K2-Think, a helpful assistant trained by MBZUAI. To answer the user's question, you first think about the reasoning process and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."
    SOLVER_TEMPERATURE: float = 1.0


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
env = Env()


# Schema for structured output
class BoNIndex(BaseModel):
    index: int  # must be 0 or 1
    explanation: str

class QuestionList(BaseModel):
    questions: list[str]

class SearchList(BaseModel):
    is_hard_problem: bool
    plan: str
    search_list: list[str]

class K2ThinkPipeline:
    def __init__(self):
        self.solver_llm = AsyncOpenAI(
            api_key=env.SOLVER_LLM_API_KEY,
            base_url=env.SOLVER_LLM_BASE_URL,
            timeout=None
        )
        self.planner_llm = AsyncOpenAI(
            api_key=env.PLANNER_LLM_API_KEY,
            base_url=env.PLANNER_LLM_BASE_URL
        )
        self.bon_responses = {}

    async def run(self, question: str) -> ChatCompletion:
        return await self.best_of_n_sampling(question=question, n=3, timeout=1200)

    async def run_at_least_one(
        self,
        fn: Callable[[], Awaitable[Any]],
        args_list:List[Any] = [], 
        soft_timeout:float = 540,
        hard_timeout:float = 3600 * 2,
        poll_interval:float = 10
    ) -> List[Any]:
        start_time = time.monotonic()
        futures = [asyncio.ensure_future(fn(*args)) for args in args_list]
        pending = set(futures)

        is_first_iteration = True

        try:
            while pending:
                # Adjust soft timeout based on how much time has passed
                elapsed = time.monotonic() - start_time
                if elapsed >= hard_timeout:
                    break
                if is_first_iteration:
                    is_first_iteration = False
                    timeout = soft_timeout
                    return_when = asyncio.ALL_COMPLETED
                else:
                    timeout = min(poll_interval, hard_timeout - elapsed)
                    return_when = asyncio.FIRST_COMPLETED
                done, pending = await asyncio.wait(
                    pending,
                    timeout=timeout,
                    return_when=return_when
                )
                if len(done) == 0:
                    continue
                results = []

                for fut in done:
                    try:
                        result = fut.result()
                        results.append(result)
                    except Exception as e:
                        log.error(f"Error in getting result: {e}")
                        continue  # Ignore failed tasks
                if len(results) > 0:
                    # Cancel the rest
                    try:
                        for p in pending:
                            p.cancel()
                    except Exception as e:
                        log.error(f"Error in canceling tasks: {e}")
                    return results
        finally:
            for fut in pending:
                fut.cancel()

        raise asyncio.TimeoutError(f"No task succeeded within hard timeout of {hard_timeout} seconds")

    async def select_best(self, question, completions):
        answers = []
        for completion in completions:
            if completion is None:
                answers.append("No answer.")
                continue
            content = completion.choices[0].message.content
            if "</think>" in content:
                answers.append(content.split("</think>")[1])
            else:
                answers.append(f"No answer was found, but here was the tail end of the problem solving: {content[-2000:]}")

        best_index, best_completion, best_answer = 0, completions[0], answers[0]
        for index, completion in enumerate(completions[1:], start=1):
            response = await self.planner_llm.chat.completions.create(
                model=env.PLANNER_LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict evaluator. Given a question and two responses, "
                            "return a JSON object with 'better_index' as 0 or 1 for the response "
                            "that best answers the question."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Question: {question}\nResponse 0: {best_answer}\nResponse 1: {answers[index]}"
                    }
                ],
                extra_body={"guided_json": BoNIndex.model_json_schema()},
            )
            winner = json.loads(response.choices[0].message.content)["index"]
            if winner == 1:
                best_index = index
                best_completion = completion
                best_answer = answers[index]
        return best_index, best_completion

    async def best_of_n_sampling(self, question: str, n: int = 3, timeout: float = 540) -> ChatCompletion | None:
        request_id = uuid.uuid4()
        self.bon_responses[request_id] = {
            "completions": [None] * n
        }
        args_list = [
            (request_id, bon_id, question)
            for bon_id in range(n)
        ]
        log.info(f"running {len(args_list)} tasks , {self.single_sampling.__name__}, {args_list}, {timeout}")
        try:
            results = await self.run_at_least_one(self.single_sampling, args_list, timeout)
            log.info(f"{results=}")
        except Exception as e:
            log.error(f"Error in best_of_n_sampling: {e}")
            return None

        completions = self.bon_responses[request_id]["completions"]
        best_index, best_completion = await self.select_best(question, completions)
        log.info(f"{best_index=}")
        return best_completion

    async def single_sampling(self, request_id: uuid.UUID, bon_id: int, question: str):
        # Get a single completion
        response = await self.sampling_with_planning(question)
        self.bon_responses[request_id]["completions"][bon_id] = response

    async def sampling_with_planning(self, question: str):

        topics_list = await self.create_topics_list(question)

        ideas = None
        if topics_list is not None:

            prompt_planning_topics: str = f'''
You are given a question and some useful topics:
<question>{question}</question>
<topics>{topics_list}</topics>
You need to generate a plan of solving the question based on the topics above WITHOUT disclosing the final or potential answer. DO NOT mention or give any hints of the final or potential answer in your plan. Wrap your plan inside <plan></plan>.
'''.strip('\n')

            response = await self.planner_llm.chat.completions.create(
                model=env.PLANNER_LLM_MODEL,
                messages=[{"role": "user", "content": prompt_planning_topics}],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            )
            ideas = response.choices[0].message.content
            if "</plan>" in ideas:
                ideas = ideas.split("</plan>")[-2]
                if "<plan>" in ideas:
                    ideas = ideas.split("<plan>")[-1]

            prompt: str = f"<question>{question}</question>" + f'''
Below are some helpful insights or ideas:
<ideas>{ideas}</ideas>
The ideas above may provide some insights in solving the challenge. Now please answer the original question.
'''.strip('\n')
        else:
            prompt = question
        response = await self.solver_llm.chat.completions.create(
            model=env.SOLVER_LLM_MODEL,
            messages=[
                {"role": "system", "content": env.SOLVER_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=env.SOLVER_TEMPERATURE
        )
        return response

    async def create_topics_list(self, question: str):

        json_schema = SearchList.model_json_schema()

        completion = await self.planner_llm.chat.completions.create(
            model=env.PLANNER_LLM_MODEL,
            messages=[
                   {"role": "user", "content": (
                            "First determine if the user is asking a hard math, stem, or coding problem, or any question where you would need more information from the internet. If so, construct a plan and then a JSON list of less than five things you would want to search for to help solve this hard problem: "
                            f'{question} '
                            'An example of a good thing to search is "Prime vs. composite power sum convergence conditions" or "Using inclusion/exclusion to prove combinatorics problems"'
                    )}
            ],
            extra_body={"guided_json": json_schema},
        )
        body = json.loads(completion.choices[0].message.content)
        if body["is_hard_problem"]:
            topics_list = body["search_list"]
        else:
            topics_list = None
        return topics_list


async def main(query: str):
    pipeline = K2ThinkPipeline()
    return await pipeline.run(query)


if __name__ == "__main__":
    query: str = "Determine the least real number $M$ such that the inequality \\[|ab(a^{2}-b^{2})+bc(b^{2}-c^{2})+ca(c^{2}-a^{2})| \\leq M(a^{2}+b^{2}+c^{2})^{2}\\] holds for all real numbers $a$, $b$ and $c$."
    response = asyncio.run(main(query))
    log.info(response)
