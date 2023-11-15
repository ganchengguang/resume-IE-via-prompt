# Please add following code in the OpenPrompt Library Sourcecode. Because we need custom the OpenPrompt to adapt resume dataset. So let Openprompt can import and train from resume dataset.

#This code is for format of seven-class resume dataset classification task.
#
#
# replace following code from     site-packages\openprompt\data_utils\text_classification_dataset.py

class resumeProcessor(DataProcessor):
    """
    `AG News <https://arxiv.org/pdf/1509.01626.pdf>`_ is a News Topic classification dataset

    we use dataset provided by `LOTClass <https://github.com/yumeng5/LOTClass>`_

    Examples:

    ..  code-block:: python

        from openprompt.data_utils.text_classification_dataset import PROCESSORS

        base_path = "datasets/TextClassification"

        dataset_name = "agnews"
        dataset_path = os.path.join(base_path, dataset_name)
        processor = PROCESSORS[dataset_name.lower()]()
        trainvalid_dataset = processor.get_train_examples(dataset_path)
        test_dataset = processor.get_test_examples(dataset_path)

        assert processor.get_num_labels() == 4
        assert processor.get_labels() == ["World", "Sports", "Business", "Tech"]
        assert len(trainvalid_dataset) == 120000
        assert len(test_dataset) == 7600
        assert test_dataset[0].text_a == "Fears for T N pension after talks"
        assert test_dataset[0].text_b == "Unions representing workers at Turner   Newall say they are 'disappointed' after talks with stricken parent firm Federal Mogul."
        assert test_dataset[0].label == 2
    """

    def __init__(self):
        super().__init__()
        self.labels = ["Personal Information", "Experience", "Summary", "Education","Qualification Certification", "Skill","Object"]

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.csv".format(split))
        examples = []
        with open(path, encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, headline = row
                text_a = headline.replace('\\', ' ')
                # text_b = body.replace('\\', ' ')
                # print(label)
                example = InputExample(guid=str(idx), text_a=text_a,  label=int(label)-1)
                examples.append(example)
        return examples



#replace following code from     site-packages/openprompt/data_utils/data_processor.py
    def get_train_examples(self, data_dir: Optional[str] = None) -> InputExample:
        """
        get train examples from the training file under :obj:`data_dir`

        call ``get_examples(data_dir, "train")``, see :py:meth:`~openprompt.data_utils.data_processor.DataProcessor.get_examples`
        """
        return self.get_examples(data_dir, "train") # this train is you train dataset file name


    def get_test_examples(self, data_dir: Optional[str] = None) -> List[InputExample]:
        """
        get test examples from the test file under :obj:`data_dir`

        call ``get_examples(data_dir, "test")``, see :py:meth:`~openprompt.data_utils.data_processor.DataProcessor.get_examples`
        """
        return self.get_examples(data_dir, "test") # this test is you test dataset file name






#
#
#
#  replace following code from         site-packages/openprompt/data_utils/utils.py
class InputExample(object):
    """A raw input example consisting of segments of text,
    a label for classification task or a target sequence of generation task.
    Other desired information can be passed via meta.

    Args:
        guid (:obj:`str`, optional): A unique identifier of the example.
        text_a (:obj:`str`, optional): The placeholder for sequence of text.
        text_b (:obj:`str`, optional): A secend sequence of text, which is not always necessary.
        label (:obj:`int`, optional): The label id of the example in classification task.
        tgt_text (:obj:`Union[str,List[str]]`, optional):  The target sequence of the example in a generation task..
        meta (:obj:`Dict`, optional): An optional dictionary to store arbitrary extra information for the example.
    """

    def __init__(self,
                 guid = None,
                 text_a = "",
                #  text_b = "",
                 label = None,
                 meta: Optional[Dict] = None,
                 tgt_text: Optional[Union[str,List[str]]] = None
                ):

        self.guid = guid
        self.text_a = text_a
        # self.text_b = text_b
        self.label = label
        self.meta = meta if meta else {}
        self.tgt_text = tgt_text

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        r"""Serialize this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        r"""Serialize this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def keys(self, keep_none=False):
        return [key for key in self.__dict__.keys() if getattr(self, key) is not None]

    @staticmethod
    def load_examples(path: str) -> List['InputExample']:
        """Load a set of input examples from a file"""
        with open(path, 'rb') as fh:
            return pickle.load(fh)

    @staticmethod
    def save_examples(examples: List['InputExample'], path: str) -> None:
        """Save a set of input examples to a file"""
        with open(path, 'wb') as fh:
            pickle.dump(examples, fh)








#
#
#
#  replace following code from       site-packages/openprompt/prompts/manual_template.py
class resumeManualTemplate(Template):
    """
    Args:
        tokenizer (:obj:`PreTrainedTokenizer`): A tokenizer to appoint the vocabulary and the tokenization strategy.
        text (:obj:`Optional[List[str]]`, optional): manual template format. Defaults to None.
        placeholder_mapping (:obj:`dict`): A place holder to represent the original input text. Default to ``{'<text_a>': 'text_a', '<text_b>': 'text_b'}``
    """

    registered_inputflag_names = ["loss_ids", 'shortenable_ids']

    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 text: Optional[str] = None,
                 placeholder_mapping: dict = {'<text_a>':'text_a'},
                ):
        super().__init__(tokenizer=tokenizer,
                         placeholder_mapping=placeholder_mapping)
        self.text = text

    def on_text_set(self):
        """
        when template text was set

        1. parse text
        """

        self.text = self.parse_text(self.text)





