from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

# 1. THE STATE
class ResumeState(TypedDict):
    resume_text: str
    feedback: str
    ats_score: int
    loop_count: int

# Setup LLM
llm = ChatGroq(model="llama-3.1-8b-instant", api_key="YOUR_API_KEY")

# We use Pydantic to force the LLM to reply with a strict JSON format for the grader
class GradeOutput(BaseModel):
    score: int = Field(description="The ATS score from 0 to 100")
    feedback: str = Field(description="Specific feedback on what is missing or needs improvement")

# 2. THE NODES
def optimizer(state: ResumeState):
    print(f"\n--- OPTIMIZING RESUME (Iteration {state.get('loop_count', 0) + 1}) ---")

    if state.get("feedback"):
        prompt = f"Improve this resume based on the following feedback.\n\nFeedback: {state['feedback']}\n\nResume:\n{state['resume_text']}"
    else:
        prompt = f"Format and enhance this raw resume with strong action verbs:\n{state['resume_text']}"

    response = llm.invoke(prompt)

    return {
        "resume_text": response.content,
        "loop_count": state.get("loop_count", 0) + 1
    }

def grader(state: ResumeState):
    print("--- GRADING RESUME ---")
    prompt = f"Grade this resume out of 100 for ATS compatibility. Be strict.\n\nResume:\n{state['resume_text']}"

    # .with_structured_output forces the LLM to return our Pydantic class
    grader_llm = llm.with_structured_output(GradeOutput)
    result = grader_llm.invoke(prompt)

    print(f"Score: {result.score}/100")
    print(f"Feedback: {result.feedback}")

    return {
        "ats_score": result.score,
        "feedback": result.feedback
    }

# 3. THE CONDITIONAL ROUTING FUNCTION
def route_resume(state: ResumeState):
    score = state.get("ats_score", 0)
    loops = state.get("loop_count", 0)

    # If the score is high enough, stop.
    if score >= 85:
        print("\n-> SUCCESS: ATS Score is excellent. Finishing up.")
        return "end"

    # Safety mechanism: Stop after 3 tries so we don't get stuck in an infinite loop
    elif loops >= 3:
        print("\n-> WARNING: Max iterations reached. Ending to prevent infinite loop.")
        return "end"

    # Otherwise, loop back!
    else:
        print("\n-> ROUTING: Score too low. Sending back to optimizer.")
        return "optimize"

# 4. WIRING THE GRAPH
def main():
    builder = StateGraph(ResumeState)

    builder.add_node("optimizer", optimizer)
    builder.add_node("grader", grader)

    # Standard edges
    builder.add_edge(START, "optimizer")
    builder.add_edge("optimizer", "grader")

    # THE CONDITIONAL EDGE
    # Syntax: add_conditional_edges(source_node, routing_function, path_map)
    builder.add_conditional_edges(
        "grader",
        route_resume,
        {
            "optimize": "optimizer", # If route_resume returns "optimize", go to the optimizer node
            "end": END               # If route_resume returns "end", stop the graph
        }
    )

    graph = builder.compile()

    # The starting data
    initial_state = {
        "resume_text": "Sutharsan N.\nPython Developer.\nWorked on Gen AI and trained ML/DL models. Did some data stuff.",
        "loop_count": 0,
        "ats_score": 0,
        "feedback": ""
    }

    final_state = graph.invoke(initial_state)
    print("\n=== FINAL RESUME ===")
    print(final_state["resume_text"])

if __name__ == "__main__":
    main()
