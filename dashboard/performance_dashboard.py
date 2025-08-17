#!/usr/bin/env python3
"""
TOPCART Performance Dashboard
============================

Real-time dashboard showing performance comparison:
- Raw LLM performance (baseline)
- TOPCART-enhanced LLM performance
- Clear performance differences and improvements

This provides transparent, verifiable evidence of our system's value.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Any
import asyncio

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from topological_cartesian.ollama_integration import OllamaLLMIntegrator, HybridCoordinateLLM
from topological_cartesian.coordinate_engine import EnhancedCoordinateEngine

class TOPCARTDashboard:
    """
    Interactive dashboard for TOPCART performance comparison.
    
    Shows real-time comparison between:
    - Raw LLM performance (baseline)
    - TOPCART-enhanced performance (our system)
    """
    
    def __init__(self):
        self.setup_page()
        self.coordinate_engine = None
        self.available_models = []
        
    def setup_page(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title="TOPCART Performance Dashboard",
            page_icon="🎯",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
        }
        .improvement-positive {
            color: #28a745;
            font-weight: bold;
        }
        .improvement-negative {
            color: #dc3545;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown('<h1 class="main-header">🎯 TOPCART Performance Dashboard</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        **Real-time comparison of AI performance:**
        - 📊 **Baseline**: Raw LLM performance (no enhancement)
        - 🚀 **TOPCART**: Enhanced with Topological Cartesian system
        - 📈 **Improvement**: Clear performance differences
        """)
        
        st.divider()
    
    def render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.header("🎛️ Test Configuration")
        
        # Model selection
        if not self.available_models:
            self.available_models = self.get_available_models()
        
        selected_model = st.sidebar.selectbox(
            "Select LLM Model",
            self.available_models,
            index=0 if self.available_models else None
        )
        
        # Test type selection
        test_type = st.sidebar.selectbox(
            "Select Test Type",
            ["GSM8K Math Problems", "General Q&A", "Code Generation", "Custom Query"]
        )
        
        # Number of problems
        num_problems = st.sidebar.slider(
            "Number of Test Problems",
            min_value=1,
            max_value=10,
            value=3
        )
        
        # Custom query input
        custom_query = ""
        if test_type == "Custom Query":
            custom_query = st.sidebar.text_area(
                "Enter Custom Query",
                placeholder="Enter your test query here..."
            )
        
        return selected_model, test_type, num_problems, custom_query
    
    def get_available_models(self) -> List[str]:
        """Get list of available Ollama models"""
        try:
            integrator = OllamaLLMIntegrator()
            return integrator.get_available_models()
        except Exception as e:
            st.error(f"Failed to get models: {e}")
            return ["llama3.2:3b", "mistral:latest"]  # Fallback
    
    def initialize_systems(self, model: str):
        """Initialize both baseline and TOPCART systems"""
        try:
            # Initialize coordinate engine if not already done
            if self.coordinate_engine is None:
                with st.spinner("Initializing TOPCART coordinate engine..."):
                    self.coordinate_engine = EnhancedCoordinateEngine()
            
            # Initialize LLM integrator
            ollama = OllamaLLMIntegrator(default_model=model)
            
            # Initialize hybrid system
            hybrid = HybridCoordinateLLM(self.coordinate_engine, ollama)
            
            return ollama, hybrid
            
        except Exception as e:
            st.error(f"Failed to initialize systems: {e}")
            return None, None
    
    def run_comparison_test(self, model: str, test_type: str, num_problems: int, custom_query: str = ""):
        """Run comparison test between baseline and TOPCART"""
        
        # Initialize systems
        baseline_llm, topcart_system = self.initialize_systems(model)
        if not baseline_llm or not topcart_system:
            return None
        
        # Get test problems
        test_problems = self.get_test_problems(test_type, num_problems, custom_query)
        
        results = {
            "model": model,
            "test_type": test_type,
            "problems": [],
            "baseline_stats": {"correct": 0, "total_time": 0.0, "avg_time": 0.0},
            "topcart_stats": {"correct": 0, "total_time": 0.0, "avg_time": 0.0}
        }
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, problem in enumerate(test_problems):
            progress = (i + 1) / len(test_problems)
            progress_bar.progress(progress)
            status_text.text(f"Testing problem {i+1}/{len(test_problems)}: {problem['question'][:50]}...")
            
            problem_result = {
                "id": i + 1,
                "question": problem["question"],
                "correct_answer": problem.get("answer", "N/A"),
                "baseline": {},
                "topcart": {}
            }
            
            # Test baseline LLM
            try:
                start_time = time.time()
                baseline_response = baseline_llm.generate(
                    prompt=self.create_test_prompt(problem["question"]),
                    model=model,
                    temperature=0.1
                )
                baseline_time = time.time() - start_time
                
                baseline_answer = self.extract_answer(baseline_response, test_type)
                baseline_correct = self.check_correctness(baseline_answer, problem.get("answer"), test_type)
                
                problem_result["baseline"] = {
                    "response": baseline_response[:200] + "..." if len(baseline_response) > 200 else baseline_response,
                    "answer": baseline_answer,
                    "correct": baseline_correct,
                    "time": baseline_time
                }
                
                if baseline_correct:
                    results["baseline_stats"]["correct"] += 1
                results["baseline_stats"]["total_time"] += baseline_time
                
            except Exception as e:
                problem_result["baseline"] = {"error": str(e), "correct": False, "time": 0.0}
            
            # Test TOPCART system
            try:
                start_time = time.time()
                topcart_result = topcart_system.process_query(
                    query=self.create_test_prompt(problem["question"]),
                    model=model,
                    temperature=0.1,
                    max_context_docs=3
                )
                topcart_time = time.time() - start_time
                
                topcart_response = topcart_result.get("llm_response", "")
                topcart_answer = self.extract_answer(topcart_response, test_type)
                topcart_correct = self.check_correctness(topcart_answer, problem.get("answer"), test_type)
                
                problem_result["topcart"] = {
                    "response": topcart_response[:200] + "..." if len(topcart_response) > 200 else topcart_response,
                    "answer": topcart_answer,
                    "correct": topcart_correct,
                    "time": topcart_time,
                    "coordinate_context": len(topcart_result.get("coordinate_context", []))
                }
                
                if topcart_correct:
                    results["topcart_stats"]["correct"] += 1
                results["topcart_stats"]["total_time"] += topcart_time
                
            except Exception as e:
                problem_result["topcart"] = {"error": str(e), "correct": False, "time": 0.0}
            
            results["problems"].append(problem_result)
        
        # Calculate averages
        if len(test_problems) > 0:
            results["baseline_stats"]["avg_time"] = results["baseline_stats"]["total_time"] / len(test_problems)
            results["topcart_stats"]["avg_time"] = results["topcart_stats"]["total_time"] / len(test_problems)
        
        progress_bar.progress(1.0)
        status_text.text("✅ Testing completed!")
        
        return results
    
    def get_test_problems(self, test_type: str, num_problems: int, custom_query: str = "") -> List[Dict]:
        """Get test problems based on type"""
        
        if test_type == "Custom Query":
            return [{"question": custom_query, "answer": "N/A"}]
        
        elif test_type == "GSM8K Math Problems":
            # Load from GSM8K dataset
            problems = []
            try:
                with open("gsm8k_test.jsonl", "r") as f:
                    for i, line in enumerate(f):
                        if i >= num_problems:
                            break
                        data = json.loads(line)
                        # Extract numerical answer
                        import re
                        answer_match = re.search(r'####\s*([0-9,]+(?:\.[0-9]+)?)', data['answer'])
                        numerical_answer = float(answer_match.group(1).replace(',', '')) if answer_match else 0.0
                        
                        problems.append({
                            "question": data["question"],
                            "answer": numerical_answer
                        })
            except FileNotFoundError:
                # Fallback problems
                problems = [
                    {
                        "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
                        "answer": 18.0
                    },
                    {
                        "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
                        "answer": 3.0
                    }
                ][:num_problems]
            
            return problems
        
        elif test_type == "General Q&A":
            return [
                {"question": "What is the capital of France?", "answer": "Paris"},
                {"question": "Who wrote Romeo and Juliet?", "answer": "William Shakespeare"},
                {"question": "What is 2 + 2?", "answer": "4"}
            ][:num_problems]
        
        elif test_type == "Code Generation":
            return [
                {"question": "Write a Python function to calculate factorial", "answer": "def factorial(n):"},
                {"question": "Create a function to reverse a string", "answer": "def reverse_string(s):"}
            ][:num_problems]
        
        return []
    
    def create_test_prompt(self, question: str) -> str:
        """Create standardized test prompt"""
        return f"""
Question: {question}

Please provide a clear, accurate answer. For math problems, show your work and provide the final numerical answer.

Answer:"""
    
    def extract_answer(self, response: str, test_type: str) -> str:
        """Extract answer from response based on test type"""
        if test_type == "GSM8K Math Problems":
            # Extract numerical answer
            import re
            # Look for final answer patterns
            patterns = [
                r'FINAL ANSWER:\s*([0-9,]+(?:\.[0-9]+)?)',
                r'\$([0-9,]+(?:\.[0-9]+)?)',
                r'([0-9,]+(?:\.[0-9]+)?)\s*dollars?',
                r'=\s*([0-9,]+(?:\.[0-9]+)?)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    try:
                        return float(match.group(1).replace(',', ''))
                    except ValueError:
                        continue
            
            # Fallback: last number in response
            numbers = re.findall(r'([0-9,]+(?:\.[0-9]+)?)', response)
            if numbers:
                try:
                    return float(numbers[-1].replace(',', ''))
                except ValueError:
                    pass
            
            return 0.0
        
        else:
            # For other types, return first line of response
            return response.split('\n')[0].strip()[:100]
    
    def check_correctness(self, predicted, expected, test_type: str) -> bool:
        """Check if answer is correct"""
        if test_type == "GSM8K Math Problems":
            try:
                return abs(float(predicted) - float(expected)) < 0.01
            except (ValueError, TypeError):
                return False
        else:
            # Simple string matching for other types
            return str(predicted).lower().strip() in str(expected).lower().strip()
    
    def render_results(self, results: Dict):
        """Render comparison results"""
        if not results:
            return
        
        st.header("📊 Performance Comparison Results")
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        baseline_accuracy = results["baseline_stats"]["correct"] / len(results["problems"]) * 100
        topcart_accuracy = results["topcart_stats"]["correct"] / len(results["problems"]) * 100
        accuracy_improvement = topcart_accuracy - baseline_accuracy
        
        with col1:
            st.metric(
                "Baseline Accuracy",
                f"{baseline_accuracy:.1f}%",
                f"{results['baseline_stats']['correct']}/{len(results['problems'])}"
            )
        
        with col2:
            st.metric(
                "TOPCART Accuracy",
                f"{topcart_accuracy:.1f}%",
                f"{results['topcart_stats']['correct']}/{len(results['problems'])}"
            )
        
        with col3:
            st.metric(
                "Improvement",
                f"{accuracy_improvement:+.1f}%",
                "Better" if accuracy_improvement > 0 else "Worse" if accuracy_improvement < 0 else "Same"
            )
        
        # Performance charts
        self.render_performance_charts(results)
        
        # Detailed results
        self.render_detailed_results(results)
    
    def render_performance_charts(self, results: Dict):
        """Render performance visualization charts"""
        
        # Accuracy comparison
        fig_accuracy = go.Figure(data=[
            go.Bar(name='Baseline', x=['Accuracy'], y=[results["baseline_stats"]["correct"] / len(results["problems"]) * 100]),
            go.Bar(name='TOPCART', x=['Accuracy'], y=[results["topcart_stats"]["correct"] / len(results["problems"]) * 100])
        ])
        fig_accuracy.update_layout(title="Accuracy Comparison", yaxis_title="Accuracy (%)")
        
        # Time comparison
        fig_time = go.Figure(data=[
            go.Bar(name='Baseline', x=['Avg Time'], y=[results["baseline_stats"]["avg_time"]]),
            go.Bar(name='TOPCART', x=['Avg Time'], y=[results["topcart_stats"]["avg_time"]])
        ])
        fig_time.update_layout(title="Response Time Comparison", yaxis_title="Time (seconds)")
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_accuracy, use_container_width=True)
        with col2:
            st.plotly_chart(fig_time, use_container_width=True)
    
    def render_detailed_results(self, results: Dict):
        """Render detailed problem-by-problem results"""
        st.subheader("📋 Detailed Results")
        
        for problem in results["problems"]:
            with st.expander(f"Problem {problem['id']}: {problem['question'][:50]}..."):
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🔹 Baseline LLM**")
                    baseline = problem["baseline"]
                    if "error" not in baseline:
                        st.write(f"Answer: {baseline['answer']}")
                        st.write(f"Correct: {'✅' if baseline['correct'] else '❌'}")
                        st.write(f"Time: {baseline['time']:.2f}s")
                        st.text_area("Response:", baseline["response"], height=100, key=f"baseline_{problem['id']}")
                    else:
                        st.error(f"Error: {baseline['error']}")
                
                with col2:
                    st.markdown("**🚀 TOPCART Enhanced**")
                    topcart = problem["topcart"]
                    if "error" not in topcart:
                        st.write(f"Answer: {topcart['answer']}")
                        st.write(f"Correct: {'✅' if topcart['correct'] else '❌'}")
                        st.write(f"Time: {topcart['time']:.2f}s")
                        st.write(f"Context Docs: {topcart.get('coordinate_context', 0)}")
                        st.text_area("Response:", topcart["response"], height=100, key=f"topcart_{problem['id']}")
                    else:
                        st.error(f"Error: {topcart['error']}")
    
    def run_dashboard(self):
        """Main dashboard execution"""
        self.render_header()
        
        # Sidebar controls
        selected_model, test_type, num_problems, custom_query = self.render_sidebar()
        
        # Main content
        if st.button("🚀 Run Performance Comparison", type="primary"):
            if selected_model:
                with st.spinner("Running comparison test..."):
                    results = self.run_comparison_test(selected_model, test_type, num_problems, custom_query)
                    if results:
                        self.render_results(results)
                        
                        # Save results
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = f"dashboard_results_{timestamp}.json"
                        with open(filename, 'w') as f:
                            json.dump(results, f, indent=2)
                        st.success(f"Results saved to {filename}")
            else:
                st.error("Please select a model first")
        
        # Information section
        st.markdown("---")
        st.markdown("""
        ### 🎯 About TOPCART Performance Dashboard
        
        This dashboard provides **transparent, real-time comparison** between:
        - **Baseline**: Raw LLM performance without any enhancement
        - **TOPCART**: Same LLM enhanced with our Topological Cartesian system
        
        **Key Features**:
        - ✅ **Side-by-side comparison** on identical problems
        - ✅ **Multiple test types** (math, Q&A, code, custom)
        - ✅ **Real-time results** with detailed analysis
        - ✅ **Transparent methodology** - see exactly what's tested
        - ✅ **Reproducible results** - save and share test results
        
        **Why This Matters**:
        - 🔬 **Scientific rigor**: Controlled comparison eliminates bias
        - 📊 **Quantified improvement**: Exact performance differences
        - 🎯 **Transparent evidence**: Anyone can verify our claims
        - 🚀 **Commercial validation**: Clear value proposition
        """)

def main():
    """Run the TOPCART Performance Dashboard"""
    dashboard = TOPCARTDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()