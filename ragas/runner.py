from autorag.deploy import Runner
import pandas as pd
from typing import List, Dict, Optional, Union
from autorag.utils.util import to_list
from autorag.schema.metricinput import MetricInput
from autorag.evaluation import evaluate_generation
from autorag.evaluation.util import cast_metrics


def evaluate_generator_node(
    result_df: pd.DataFrame,
    metric_inputs: List[MetricInput],
    metrics: Union[List[str], List[Dict]],
):
    @evaluate_generation(metric_inputs=metric_inputs, metrics=metrics)
    def evaluate_generation_module(df: pd.DataFrame):
        return (
            df["generated_texts"].tolist(),
            df["generated_tokens"].tolist(),
            df["generated_log_probs"].tolist(),
        )

    return evaluate_generation_module(result_df)

class AdvancedRunner(Runner):
    def __init__(self, config: Dict, project_dir: Optional[str] = None):
        super().__init__(config, project_dir)

    def run_with_qa(self, qa_data: pd.DataFrame, strategy: Dict) -> pd.DataFrame:
        """
        Run the pipeline with qa_data.
        The loaded pipeline must start with a single query,
        so the first module of the pipeline must be `query_expansion` or `retrieval` module.

        qa_data: pd.DataFrame(
            {
                "qid": str(uuid.uuid4()),
                "query": [query],
                "retrieval_gt": [[]],
                "generation_gt": [""],
            }
        )
        :return: The result of the pipeline.
        """
        assert qa_data is not None, "qa_data must not be None"
        # Init Metrics: make rows to metric_inputs
        generation_gt = to_list(qa_data["generation_gt"].tolist())
        metric_inputs = [MetricInput(generation_gt=gen_gt) for gen_gt in generation_gt]
        previous_result = qa_data

        for i, (module_instance, module_param) in enumerate(zip(
                self.module_instances, self.module_params
        )):
            print(module_instance, module_param)
            new_result = module_instance.pure(
                previous_result=previous_result, **module_param
            )

            # Evaluate the results if arriving the last module (the module is a generator)
            if i == len(self.module_instances) - 1:
                """
                The format of new_result:
                [''generated_texts', 'generated_tokens', 'generated_log_probs', 'meteor', 'rouge', 'bert_score']
                """
                new_result = evaluate_generator_node(new_result, metric_inputs, strategy.get("metrics"))

            duplicated_columns = previous_result.columns.intersection(
                new_result.columns
            )
            drop_previous_result = previous_result.drop(columns=duplicated_columns)
            previous_result = pd.concat([drop_previous_result, new_result], axis=1)

        print("Results content:\n", previous_result.head())
        print("Column names:\n", previous_result.columns)
        print("Example of a full row:\n", previous_result.iloc[0])
        return previous_result