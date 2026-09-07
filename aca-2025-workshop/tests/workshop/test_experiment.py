import unittest
from src.rag.experiment import prepare_prompt, validate_training, score_output
from test_retrieval import doc

class Tokenizer:
    def encode(self,s,**kw):return list(s)
    def apply_chat_template(self,messages,**kw):return ''.join(m['content'] for m in messages)

class ExperimentTests(unittest.TestCase):
    def test_prompt_parity_and_reference_exclusion(self):
        config=dict(max_input_tokens=1800,max_new_tokens=100)
        q={'id':'q1','question':'What is covered?','filters':{'namespace':'plans'},'reference_answer':'SECRET GOLD'}
        prompt,meta=prepare_prompt(q,[doc('a','delivery covered')],Tokenizer(),config,policy='strict')
        self.assertNotIn('SECRET GOLD',str(prompt));self.assertIn('a',str(prompt))
        self.assertLessEqual(meta['input_tokens'],1800)

    def test_invalid_and_fabricated_citations(self):
        self.assertFalse(score_output('{"answer":"yes","action":"answer","citations":["fake"]}',{'a'})['citations_valid'])
        self.assertFalse(score_output('not json',{'a'})['valid_json'])

    def test_training_split_overlap(self):
        row=dict(id='a',question='q',messages=[])
        with self.assertRaises(ValueError):validate_training([row],[row],[])

class LabelIsolationTests(unittest.TestCase):
    def test_filters_do_not_use_gold_numeric_keys(self):
        from src.rag.evaluate import questions_from_rows
        row=dict(id='x',namespace='enrollment',question='How many selections in Florida?',expected_values={'KY.Cnsmr':'999'})
        self.assertEqual(questions_from_rows([row])[0]['filters']['state'],'FL')
