import argparse
from typing import List

import ollama
from pydantic import (
    BaseModel,
    Field,
    field_validator,
)

from datetime import datetime


class Message(BaseModel):
    role: str
    content: str


class OllamaResponse(BaseModel):
    model: str
    created_at: datetime
    message: Message
    done: bool
    total_duration: int
    load_duration: int = 0
    prompt_eval_count: int = Field(-1, validate_default=True)
    prompt_eval_duration: int
    eval_count: int
    eval_duration: int

    @field_validator("prompt_eval_count")
    @classmethod
    def validate_prompt_eval_count(cls, value: int) -> int:
        if value == -1:
            print(
                "\nWarning: prompt token count was not provided, potentially due to prompt caching. For more info, see https://github.com/ollama/ollama/issues/2068\n"
            )
            return 0  # Set default value
        return value


def run_benchmark(
    model_name: str, prompt: str, verbose: bool
) -> OllamaResponse:

    last_element = None

    if verbose:
        stream = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            stream=True,
        )
        for chunk in stream:
            print(chunk["message"]["content"], end="", flush=True)
            last_element = chunk
    else:
        last_element = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

    if not last_element:
        print("System Error: No response received from ollama")
        return None

    # with open("data/ollama/ollama_res.json", "w") as outfile:
    #     outfile.write(json.dumps(last_element, indent=4))

    return OllamaResponse.model_validate(last_element)


def nanosec_to_sec(nanosec):
    return nanosec / 1000000000


def inference_stats(model_response: OllamaResponse):
    # Use properties for calculations
    prompt_ts = model_response.prompt_eval_count / (
        nanosec_to_sec(model_response.prompt_eval_duration)
    )
    response_ts = model_response.eval_count / (
        nanosec_to_sec(model_response.eval_duration)
    )
    total_ts = (
        model_response.prompt_eval_count + model_response.eval_count
    ) / (
        nanosec_to_sec(
            model_response.prompt_eval_duration + model_response.eval_duration
        )
    )

    print(
        f"""
----------------------------------------------------
        {model_response.model}
        \tPrompt eval: {prompt_ts:.2f} t/s
        \tResponse: {response_ts:.2f} t/s
        \tTotal: {total_ts:.2f} t/s

        Stats:
        \tPrompt tokens: {model_response.prompt_eval_count}
        \tResponse tokens: {model_response.eval_count}
        \tModel load time: {nanosec_to_sec(model_response.load_duration):.2f}s
        \tPrompt eval time: {nanosec_to_sec(model_response.prompt_eval_duration):.2f}s
        \tResponse time: {nanosec_to_sec(model_response.eval_duration):.2f}s
        \tTotal time: {nanosec_to_sec(model_response.total_duration):.2f}s
----------------------------------------------------
        """
    )


def average_stats(responses: List[OllamaResponse]):
    if len(responses) == 0:
        print("No stats to average")
        return

    res = OllamaResponse(
        model=responses[0].model,
        created_at=datetime.now(),
        message=Message(
            role="system",
            content=f"Average stats across {len(responses)} runs",
        ),
        done=True,
        total_duration=sum(r.total_duration for r in responses),
        load_duration=sum(r.load_duration for r in responses),
        prompt_eval_count=sum(r.prompt_eval_count for r in responses),
        prompt_eval_duration=sum(r.prompt_eval_duration for r in responses),
        eval_count=sum(r.eval_count for r in responses),
        eval_duration=sum(r.eval_duration for r in responses),
    )
    print("Average stats:")
    inference_stats(res)


def get_benchmark_models(skip_models: List[str] = []) -> List[str]:
    models = ollama.list().get("models", [])
    model_names = [model["name"] for model in models]
    if len(skip_models) > 0:
        model_names = [
            model for model in model_names if model not in skip_models
        ]
    print(f"Evaluating models: {model_names}\n")
    return model_names


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmarks on your Ollama models."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Increase output verbosity",
        default=False,
    )
    parser.add_argument(
        "-s",
        "--skip-models",
        nargs="*",
        default=[],
        help="List of model names to skip. Separate multiple models with spaces.",
    )
    parser.add_argument(
        "-p",
        "--prompts",
        nargs="*",
        default=[
            """Location! Location! Location! SUPER Ultra-luxury community of CHAPEL CREEK! Newly built! Mirrored model HOME! Cul De Sac Huge Corner End Lot NEVER LIVED IN DARLING CUSTOM HOME in the heart of Frisco. Immediate access to SRT & DNT Top-rated Frisco ISD. Full Iron door entrance that leads to an impressive vaulted high ceiling entry. Beautiful flooring throughout. 1st-floor master bdrm with full bath and full closet. + EQUIPPED MEDIA ROOM! Plenty of space! Well thought out home. Open floor concept living-dining area w modern kitchen w stainless appliances & granite countertops. New Fridge! The community features a walking and biking trail. Seconds away from The Star, Legacy West, Stonebriar Mall, and other major retail and dining, and entertainment. 21 minutes away from DFW airport. AMAZING LOCATION IN FRISCO A+! Come be the first tenant of this home Everything is brand new The media room is on the first floor The Master bedroom is on the first floor as well! THE HOME IS READY FOR A TENANT
            Given the information in the public remarks of real estate listings, extract detailed data relevant to a professional real estate investor and present it in a JSON format. Sample Input:Welcome to this charming 3-bedroom, 2-bathroom home nestled in Kennedale ISD! This well-maintained home offers an open and spacious living area. The well-equipped kitchen boasts an electric range, dishwasher, refrigerator, microwave, and granite countertops. The covered patio provides a serene space to relax and unwind. Storage shed offers ample room for all your belongings. Brand new luxury vinyl plank flooring and fresh paint throughout. Prime location and desirable features! Pets allowed on a case by case basis - no aggressive breeds - Monthly pet fee -$40.00 in addition to rent and pet deposit. Application fee $65.00 per adult 18 and over - NON-REFUNDABLE - copy of DL's and 2 months proof of income submitted with each application-renter's insurance required. Pet Screening must be completed - Property may be eligible for Rhino, an affordable monthly protection coverage that replaces upfront cash security deposits and lowers your move-in fees. For more information visit SAYRHINO! Sample Output: {    "rent_concessions": [        {            "type": "Reduced Fees",            "conditions": "Pet Screening must be completed",            "details": "Property may be eligible for Rhino, an affordable monthly protection coverage that replaces upfront cash security deposits and lowers your move-in fees. Visit SAYRHINO for more information."        }    ],    "property_details": {        "bedrooms": 3,        "bathrooms": 2,        "unique_features": [            "Electric range",            "Dishwasher",            "Refrigerator",            "Microwave",            "Granite countertops",            "Covered patio",            "Storage shed",            "Luxury vinyl plank flooring",            "Fresh paint"        ],        "proximity": []    },    "pet_policy": {        "allowed": "Pets allowed on a case by case basis",        "restrictions": ["No aggressive breeds","pet screening must be completed"],        "fees": {            "monthly_fee": "$40.00",            "deposit": true        }    },    "fees_and_charges": [        {            "name": "application_fee",            "amount": 65.00,            "details": "Application fee $65.00 per adult 18 and over - NON-REFUNDABLE - copy of DL's and 2 months proof of income submitted with each application."        },        {            "name": "pet_fee",            "amount": 40.00,            "details": "Monthly pet fee - in addition to rent and pet deposit."        },        {            "name": "security_deposit",            "amount": "Variable",            "details": "Property may be eligible for Rhino, which replaces upfront cash security deposits."        },        {            "name": "pet_deposit",            "amount": null,            "details": "Pets allowed on a case by case basis - no aggressive breeds."        }    ],    "requirements": [        {            "name": "renter_insurance",            "details": "each application-renter's insurance required."        },        {            "name": "copy_of_drivers_license",            "details": "copy of DL's"        },        {            "name": "proof_of_income",            "details": "2 months proof of income"        }    ],    "sentiment_or_emotion": {        "urgency": 0.9,        "positive_descriptors": [            "Charming",            "Prime location",            "Desirable features"        ],        "negative_descriptors": null    },    "additional_insights": [        {            "name": "neighborhood",            "details": "Kennedale ISD"        },        {            "name": "location_advantages",            "details": "Prime location"        }    ]} """,
            "Write a report on the financials of Apple Inc.",
        ],
        help="List of prompts to use for benchmarking. Separate multiple prompts with spaces.",
    )

    args = parser.parse_args()

    verbose = args.verbose
    skip_models = args.skip_models
    prompts = args.prompts
    print(
        f"\nVerbose: {verbose}\nSkip models: {skip_models}\nPrompts: {prompts}"
    )

    model_names = get_benchmark_models(skip_models)
    benchmarks = {}

    for model_name in model_names:
        responses: List[OllamaResponse] = []
        for prompt in prompts:
            if verbose:
                print(f"\n\nBenchmarking: {model_name}\nPrompt: {prompt}")
            response = run_benchmark(model_name, prompt, verbose=verbose)
            responses.append(response)

            if verbose:
                print(f"Response: {response.message.content}")
                inference_stats(response)
        benchmarks[model_name] = responses

    for model_name, responses in benchmarks.items():
        average_stats(responses)


if __name__ == "__main__":
    main()
    # Example usage:
    # python benchmark.py --verbose --skip-models aisherpa/mistral-7b-instruct-v02:Q5_K_M llama2:latest --prompts "What color is the sky" "Write a report on the financials of Microsoft"
