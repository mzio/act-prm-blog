"""
Prompts for Snorkel Agent Insurance Underwriting environment.

Adapted from the reference implementation at
https://github.com/snorkel-ai/multi-turn-insurance-underwriting-benchmark-generation

The original benchmark uses a simulated-user + assistant two-agent loop.
Here we provide all company information upfront in the user prompt so the
agent can focus on tool-based reasoning (SQL queries, guideline lookups).
"""

# Assistant system prompt (adapted from prompts_v0.json in the reference repo)
INSURANCE_SYSTEM_PROMPT = (
    "You are an assistant helping a P&C underwriter at All National Insurance, "
    "a fictional small-business insurance company. Your job is to use the provided "
    "tools to look up underwriting guidelines, query database tables, and answer "
    "the underwriter's question accurately.\n\n"
    "Key concepts:\n"
    "- LOB (Line of Business): A type of insurance coverage (e.g., property, "
    "general liability, commercial auto, workers compensation, cyber liability, "
    "business owners policy).\n"
    "- Appetite: Whether the company is willing to underwrite a given LOB for a "
    "given business. Can be 'Yes', 'No', or 'Qualified' (conditionally accepted).\n\n"
    "Instructions:\n"
    "- Use the tools to find information. Do NOT rely on prior knowledge about "
    "insurance rules — only use information from the tools.\n"
    "- Start by exploring available tables with list_tables and get_table_descriptions.\n"
    "- Use get_table_schema and get_table_data_dictionary to understand table structure.\n"
    "- Use read_query to run SQL SELECT queries against the database.\n"
    "- Use get_underwriting_guidelines to read the underwriting rules.\n"
    "- Think step by step. You have up to {max_turns} turns.\n"
    "- When you have gathered enough information, use the respond_user tool to "
    "provide your final answer with a clear rationale."
)


SYSTEM_PROMPTS = {
    "default": INSURANCE_SYSTEM_PROMPT,
}


def render_prompt(
    user_task: str,
    company_name: str,
    company_description: str,
    naics_code: str,
    annual_revenue: str,
    number_of_employees: str,
    state: str,
    lob: str,
    total_payroll: str = "",
    number_of_vehicles: str = "",
    building_construction: str = "",
    total_insured_value_property: str = "",
) -> str:
    """Render the user prompt for an insurance underwriting task.

    Provides all company information upfront so the agent can focus on
    tool-based reasoning rather than multi-turn information gathering.
    """
    lines = [
        f"Company: {company_name}",
        f"Description: {company_description}",
        f"NAICS Code: {naics_code}",
        f"Annual Revenue: {annual_revenue}",
        f"Number of Employees: {number_of_employees}",
        f"State: {state}",
    ]
    if lob:
        lines.append(f"Line of Business (LOB): {lob}")
    if total_payroll and str(total_payroll) not in ("", "nan", "None"):
        lines.append(f"Total Payroll: {total_payroll}")
    if number_of_vehicles and str(number_of_vehicles) not in ("", "nan", "None", "0"):
        lines.append(f"Number of Vehicles: {number_of_vehicles}")
    if building_construction and str(building_construction) not in ("", "nan", "None"):
        lines.append(f"Building Construction Type: {building_construction}")
    if total_insured_value_property and str(total_insured_value_property) not in (
        "",
        "nan",
        "None",
    ):
        lines.append(f"Total Insured Value (TIV): {total_insured_value_property}")

    company_info = "\n".join(lines)

    return f"Here is the company information:\n\n{company_info}\n\nQuestion: {user_task}"
