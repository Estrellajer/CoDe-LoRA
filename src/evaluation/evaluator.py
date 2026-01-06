import json
import re
import matplotlib.pyplot as plt
import numpy as np
import os
import random

class MetricBase:
    def __init__(self):
        raise NotImplementedError()
    def update(self, y_truth, y_pred):
        raise NotImplementedError()
    def get_metric(self):
        raise NotImplementedError()
    def get_last(self):
        raise NotImplementedError()

class MetricAcc(MetricBase):
    def __init__(self):
        self.scores = []
    def update(self, y_truth: str, y_pred: str):
        if y_truth == y_pred:
            self.scores.append(1)
        else:
            self.scores.append(0)
    def get_metric(self):
        if len(self.scores) == 0:
            return 0
        else:
            return sum(self.scores) / len(self.scores)
    def get_last(self):
        return self.scores[-1]
            
class MetricF1(MetricBase):
    def __init__(self):
        self.sum_TP = 0
        self.sum_FN = 0
        self.sum_FP = 0
        self.last_TP = None
        self.last_FN = None
        self.last_FP = None
    def update(self, y_truth: set, y_pred: set):
        # TP: exists in truth and in pred
        # FN: exists in truth but not in pred
        # FP: does not exist in truth but exists in pred
        self.last_TP = len(y_truth & y_pred)
        self.last_FN = len(y_truth - y_pred)
        self.last_FP = len(y_pred - y_truth)
        self.sum_TP += self.last_TP
        self.sum_FN += self.last_FN
        self.sum_FP += self.last_FP
    def get_metric(self):
        # TP + FN may be 0
        # TP + FP may be 0
        TP = self.sum_TP
        FN = self.sum_FN
        FP = self.sum_FP
        if TP + FN == 0:
            recall = 0
        else:
            recall = TP / (TP + FN)
        if TP + FP == 0:
            precision = 0
        else:
            precision = TP / (TP + FP)
        if recall + precision == 0:
            f1 = 0
        else:
            f1 = 2 * recall * precision / (recall + precision)
        self.recall = recall
        self.precision = precision
        return f1
    def get_detail(self):
        if not hasattr(self, 'recall'):
            f1 = self.get_metric()
        return f1, self.recall, self.precision
    def get_last(self):
        return self.last_TP, self.last_FN, self.last_FP

class MetricF1NA(MetricF1):
    "Special handling for relation type NA in RE"
    def update(self, y_truth: set, y_pred: set):
        self.last_TP = 0
        self.last_FN = 0
        self.last_FP = 0
        for truth in y_truth:
            if ',na,' in truth:
                pattern = re.escape(truth).replace(',na,', ',(.+),')    # All lowercase in evaluator._extract, so na not NA
                pattern = re.compile(pattern)
                pred_fail = False
                for pred in y_pred:
                    match = pattern.match(pred)
                    if match is not None and match.group(1) != 'na':     # truth: (A,NA,B); pred:(A,notNA,B)
                        pred_fail = True
                        break
                if not pred_fail:       # Only add TP when prediction does not give wrong explicit affirmation
                    self.last_TP += 1    # FP will be counted later, no need to calculate here to avoid double counting
            else:
                if truth in y_pred:
                    self.last_TP += 1
                else:
                    self.last_FN += 1
        for pred in y_pred:
            if ',na,' in pred:
                pattern = re.escape(pred).replace(',na,', ',(.+),')
                pattern = re.compile(pattern)
                pred_fail = False
                for truth in y_truth:
                    match = pattern.match(truth)
                    if match is not None and match.group(1) != 'na':    # pred: (A,NA,B); truth:(A,notNA,B)
                        pred_fail = True
                        break
                if pred_fail:
                    self.last_FP += 1
                else:
                    self.last_TP += 0    # Uncertain: for pred (A,NA,B), if truth does not contain (A,*,B), should it count as TP? Assume no, giving (A,NA,B) in prediction: no points if correct, penalty if wrong.
            else:
                if pred not in y_truth:
                    self.last_FP += 1
        self.sum_TP += self.last_TP
        self.sum_FN += self.last_FN
        self.sum_FP += self.last_FP

class AuditBase:
    def __init__(self, record_limit=16):
        # record_limit: maximum size of record, `-1` for infinite, `0` for no record
        self.record_limit = record_limit
        self.cnt = 0
        self.record = []
    def _check(self, last) -> bool:
        # must be overrided
        # return whether be recorded or not
        raise NotImplementedError()
    def _add_record(self, new_record):
        self.cnt += 1
        if self.record_limit < 0 or len(self.record) < self.record_limit:
            # record limit check
            self.record.append(new_record)
        elif os.environ.get('RANDOM_RECORD')=='1':
            # Streaming uniform sampling problem
            if random.randint(1,self.cnt) <= self.record_limit:
                idx = random.randint(0,len(self.record)-1)
                self.record[idx] = new_record
    def update(self, last):
        if self._check(last):
            new_record = {
                'json_data': last['json_data'],
                'predict': last['predict'],
                'y_truth': last['y_truth'],
                'y_pred': last['y_pred']
            }
            new_record = self._to_json_object(new_record)
            self._add_record(new_record)
    @staticmethod
    def _to_json_object(obj):
        if isinstance(obj, str) or isinstance(obj, int) or isinstance(obj, float):
            return obj
        if isinstance(obj, tuple) or isinstance(obj, list) or isinstance(obj, set):
            return [AuditBase._to_json_object(x) for x in obj]
        if isinstance(obj, dict):
            return {AuditBase._to_json_object(k): AuditBase._to_json_object(v) for k, v in obj.items()}
        else:
            raise NotImplementedError()
    def get_cnt(self):
        return self.cnt
    def get_record(self):
        return self.record
    def get_report(self):
        return {
            'count': self.cnt,
            'record': self.record
        }
    def get_name(self):
        # Default to class name, override this method if you want to customize the name
        return self.__class__.__name__

class AuditVoid(AuditBase):
    "Detect empty output"
    def _check(self, last) -> bool:
        return last['predict'].strip() == ''

class AuditLong(AuditBase):
    "Detect overly long output"
    def _check(self, last) -> bool:
        return len(last['predict']) >= 512     # Length limit can be modified as needed

class AuditInsane(AuditBase):
    "Detect gibberish"
    def _check(self, last) -> bool:
        return last['predict'].strip().lower() not in {'na', 'no relation', 'none', '[]', ''} and len(last['y_pred']) == 0    # Said something but nothing useful

class AuditBothEmpty(AuditBase):
    "Detect entries where both label and predict are empty"
    def _check(self, last) -> bool:
        return len(last['y_truth']) == 0 and len(last['y_pred']) == 0

class AuditLabelEmptyOnly(AuditBase):
    "Detect label is empty but predict is not"
    def _check(self, last) -> bool:
        return len(last['y_truth']) == 0 and len(last['y_pred']) != 0

class AuditPredEmptyOnly(AuditBase):
    "Detect predict is empty but label is not"
    def _check(self, last) -> bool:
        return len(last['y_truth']) != 0 and len(last['y_pred']) == 0
    
class AuditNA(AuditBase):
    "Detect output containing type NA, currently only for RE"
    def _check(self, last) -> bool:
        for i in last['y_pred']:    # assert isinstance(i, str)
            if ',na,' in i:
                return True
        return False

class AuditInvalid(AuditBase):
    "Detect output containing invalid label types, currently only for RE and NER"
    def _check(self, last) -> bool:
        valid_labels = EvaluatorBase._resolve_option(last['json_data']['Instance']['instruction'])
        if len(valid_labels) == 0:
            # If no option provided, ignore this audit item
            return False
        valid_labels = set(valid_labels)

        for pred in last['y_pred']:
            pred = pred.split(':')
            if len(pred) >= 2:
                label = pred[0]
                if label not in valid_labels:
                    return True
        return False

class AuditFidelity(AuditBase):
    "Detect entities not from sentence, currently only for RE and NER"
    def _check(self, last) -> bool:
        for item in last['y_pred']:
            item = item.split(':')       #   Hard to handle cases where entity or label itself contains commas
            if len(item) < 2:
                continue
            ents = item[-1].split(',')
            for ent in ents:
                if EvaluatorBase._format(ent) not in EvaluatorBase._format(last['json_data']['Instance']['sentence']):
                    return True
            return False

class AuditGoldenlabelFault(AuditBase):
    "Triples in golden label have gaps, currently only for RE"
    def _check(self, last) -> bool:
        for item in last['y_truth']:
            cnt = 0
            if len(item.split(':')) < 2:
                continue
            for i in item.split(':')[-1].split(','):
                i = i.strip()
                if i != '':
                    cnt += 1
            if cnt <= 1:
                return True
        return False

class AuditRepeat(AuditBase):
    "Detect repetition"
    def _check(self, last) -> bool:
        pattern = r'(\w{5,})\1{2,}'  # Match substrings of length >5 that appear three or more times consecutively
        match = re.search(pattern, last['predict'])
        return match is not None

class AuditRetard(AuditBase):
    "Detect errors when both are non-empty"
    def _check(self, last) -> bool:
        last_metric = last['metric']
        if hasattr(last_metric, 'last_TP'):
            if len(last['y_pred']) != 0 and len(last['y_truth']) != 0:
                return last_metric.last_TP == 0
        if hasattr(last_metric, 'scores'):
            return last_metric.scores[-1] == 0
        return False
        
class AuditWhatever(AuditBase):
    "Catch all"
    def _check(self, last) -> bool:
        return True
    
class AuditConfuseMatrix(AuditBase):
    """
    1. Detect contradictions, e.g., same entity has two different labels
        Example: [(texas, person), (texas, place)]
    2. Maintain and export confusion matrix, only for NER and RE
    """
    # The second function of this subclass deviates from the original intent of the Audit series, perhaps better placed in Metric series
    # But the latter approach would require expanding the information range that Metric can access, requiring major framework changes.
    # Perhaps all code becomes legacy this way step by step...
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.options = None
        self.options2idx = None
        self.matrix = None
        self.dataset_name = None
    def _check(self, last) -> bool:
        raise NotImplementedError()
    @staticmethod
    def _resolve(s):
        # 'A,B,C' --> 'A,C', 'B'
        # 'A,B' --> 'A', 'B'
        # 'A,B,C,D' --> None
        # Assume s has been standardized
        s = [i.strip() for i in s.split(',')]
        if len(s) == 2:
            return s[0], s[1]
        elif len(s) == 3:
            return '%s,%s'%(s[0],s[2]), s[1]
        else:
            return None
    def update(self, last):
        if self.dataset_name is None:
            self.dataset_name = last['json_data']['Dataset']
            self.options = EvaluatorBase._resolve_option(last['json_data']['Instance']['instruction'])
            if 'na' not in self.options:
                self.options.append('na')
            self.options2idx = dict()
            for idx, option in enumerate(self.options):
                self.options2idx[option] = idx
            N = len(self.options)
            self.matrix = np.zeros((N, N), dtype=np.int16)
        truth = dict()
        pred = dict()
        is_conflict = False
        for item in last['y_truth']:
            res = self._resolve(item)
            if res is not None:
                if res[0] in truth:
                    is_conflict = True
                truth[res[0]] = res[1]
        for item in last['y_pred']:
            res = self._resolve(item)
            if res is not None:
                if res[0] in pred:
                    is_conflict = True
                pred[res[0]] = res[1]
        for k in truth:
            if truth[k] not in self.options2idx:
                continue    # May be due to unexpected format parsing exception
            idx_truth = self.options2idx[truth[k]]
            if k in pred:
                if pred[k] in self.options2idx:
                    idx_pred = self.options2idx[pred[k]]
                    self.matrix[idx_truth][idx_pred] += 1
            else:
                idx_pred = self.options2idx['na']
                self.matrix[idx_truth][idx_pred] += 1
        for k in pred:
            if pred[k] not in self.options2idx:
                continue
            idx_pred = self.options2idx[pred[k]]
            if k not in truth:
                idx_truth = self.options2idx['na']
                self.matrix[idx_truth][idx_pred] += 1

        if is_conflict:
            new_record = {
                'json_data': last['json_data'],
                'predict': last['predict'],
                'y_truth': list(last['y_truth']),
                'y_pred': list(last['y_pred'])
            }
            new_record = self._to_json_object(new_record)
            self._add_record(new_record)

    def get_report(self):
        if os.environ.get('EXPORT_IMG') == '1':
            root = 'img'    # Hardcoding is bad practice, but this generates write-only temporary data that is only read by users, not by programs
            if not os.path.exists(root):
                os.mkdir(root)
            fpath = os.path.join(root, '%s.png'%self.dataset_name)
            if True:
                # Most of the time we don't care about main diagonal and na elements, mask them to reduce visual noise
                matrix = ((1-np.eye(self.matrix.shape[0])) * self.matrix).astype(np.int16)    
                na = self.options2idx['na']
                matrix[na,:]=0
                matrix[:,na]=0
            else:
                matrix = self.matrix
            self._plot_matrix(matrix, self.options, fpath, title=self.dataset_name)
        return super().get_report()
    @staticmethod
    def _plot_matrix(A, labels, fpath, title=None, min_size = 50):
        N = len(labels)
        figsize = N * min_size / 100 + 1, N * min_size / 100 + 1
        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(A, cmap='viridis')

        ax.set_xticks(np.arange(N))
        ax.set_yticks(np.arange(N))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

        for i in range(N):
            for j in range(N):
                text = ax.text(j, i, A[i, j], ha='center', va='center', color='w')
        
        if title is not None:
            ax.set_title(title)
        fig.tight_layout()
        plt.savefig(fpath)
        plt.close()
    
class EvaluatorBase:
    def __init__(self):
        self.last = dict()
        self._init_audit()
        self._init_metric()
    
    def _init_metric(self):
        # must be overrided to init self.metric
        self.metric = MetricBase()

    def _init_audit(self):
        # override if necessary
        # Override this method if you need to add other audit items or customize instantiation
        self.audit = [
            AuditVoid(),
            AuditBothEmpty(),
            AuditLabelEmptyOnly(),
            AuditPredEmptyOnly(),
            AuditLong(),
            AuditInsane(),
            AuditRepeat(),
            AuditRetard(),
            AuditWhatever()
        ]
    
    def _update_audit(self):
        # override if necessary
        for audit in self.audit:
            audit.update(self.last)

    def _extract(self, json_data, predict: str):
        # must be overrided
        # return: y_truth, y_pred
        raise NotImplementedError()

    def add(self, json_data, predict):
        """
        Can input multiple data items or a single data item.
        When inputting a single data item, json_data should be a json-like object (parsed by json.load), predict should be a single string.
        When inputting multiple data items, both json_data and predict should be lists.
        """
        assert isinstance(json_data, list) == isinstance(predict, list)

        if isinstance(json_data, list) and isinstance(predict, list):
            for i, j in zip(json_data, predict):
                self.add(i, j)
                return
        
        # add single case
        y_truth, y_pred = self._extract(json_data, predict)
        self.metric.update(y_truth, y_pred)

        # audit
        # last stores all information that may be needed for audit submission
        self.last['json_data'] = json_data
        self.last['predict'] = predict
        self.last['y_truth'] = y_truth
        self.last['y_pred'] = y_pred
        self.last['metric'] = self.metric

        self._update_audit()
    
    def get_metric(self) -> float:
        return self.metric.get_metric()

    def get_audit_report(self):
        'Get all audit item result reports, return a json-like object'
        return {
            a.get_name() : a.get_report()
            for a in self.audit
        }
    def dump_audit_report(self, fpath):
        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(self.get_audit_report(), f, indent=4, ensure_ascii=False)
    
    @staticmethod
    def _resolve_option(s):
        "s: instruction"
        option_parts = re.findall('Option:(.+?)\n', s)
        if len(option_parts) <= 0:
            return []
        option_part = option_parts[0]
        ans = [EvaluatorBase._format(x) for x in option_part.split(',')]
        return ans

    @staticmethod
    def _remove_redundant_space(s):
        # '   a  b  \t  c  \n' --> 'a b c'
        # '  kjc,  jns , ((  : ()  )  ( . )( ln  kc  a,,  ' --> 'kjc,jns,((:())(.)(ln kc a,,'
        s = ' '.join(s.split())     # Multiple whitespace characters become single space
        s = re.sub(r"\s*(,|:|\(|\)|\.|_|;|'|-)\s*", r'\1', s)   # Remove whitespace around special symbols
        return s
    
    @staticmethod
    def _format(s):
        "Comprehensive format normalization, centrally solving various format issues"
        s = EvaluatorBase._remove_redundant_space(s)
        s = s.lower()
        s = s.replace('{','').replace('}','')
        s = re.sub(',+', ',', s)
        s = re.sub('\.+', '.', s)
        s = re.sub(';+', ';', s)
        s = s.replace('’', "'")
        s = s.replace('location', 'located')
        return s
    
    @staticmethod
    def _re_item(s):
        # '   A,B,C),   (D,EF),  ,,(GH ' --> ['A,B,C', 'D,EF', 'GH']
        # ' A,B,C)  ' --> ['A,B,C']
        # Sometimes model output is missing opening left bracket or closing right bracket
        # This regex does not capture brackets, only captures content in between
        # Deprecated
        return re.findall(r'(?:^|\()([^\(\)]+?)(?:$|\))', s.strip())
    
    @staticmethod
    def _resolve_brackets(s):
        # Extract content within top-level paired brackets, return as string list, discard content outside brackets.
        # This function tolerates one missing left bracket at sentence start and one missing right bracket at sentence end (but not both)
        # 'a(b)(c(d))(' --> ['b', 'c(d)']
        ans = []
        level = 0
        last_lb_idx = None
        for idx, char in enumerate(s):
            if char == '(':
                if level == 0:
                    last_lb_idx = idx
                level += 1
            elif char == ')':
                if last_lb_idx is None and len(ans) == 0 and 0 != idx:
                    ans.append(s[0 : idx])
                if level == 1 and last_lb_idx+1 != idx:
                    ans.append(s[last_lb_idx+1 : idx])
                if level >= 1:
                    level -= 1
        if level == 1 and last_lb_idx+1 != len(s):
            ans.append(s[last_lb_idx+1:])
        return ans
    
    @staticmethod
    def _resolve_comma(s):
        # Split sentence by comma, but commas inside brackets don't count, ignore empty strings from split
        # 'a,(b,c),,d,' --> ['a', '(b,c)', 'd']
        ans = []
        level = 0
        last_comma = -1
        for idx, char in enumerate(s):
            if char == '(':
                level += 1
            elif char == ')':
                level -= 1
            elif char == ',' and level == 0 and last_comma + 1 != idx:
                ans.append(s[last_comma+1 : idx])
                last_comma = idx
        if last_comma+1 != len(s):
            ans.append(s[last_comma+1:])
        return ans

class EvaluatorNER(EvaluatorBase):
    def _init_metric(self):
        self.metric = MetricF1()

    def _init_audit(self):
        super()._init_audit()    
        self.audit += [
            AuditInvalid(),
            AuditFidelity(),
            AuditConfuseMatrix()
        ]
    def _extract(self, json_data, predict):
        # person: a; person: b; org: c
        entity_truth = set()
        for ent in self._format(json_data['Instance']['ground_truth']).split(';'):
            ent = self._format(ent)
            entity_truth.add(ent)
        
        entity_pred = set()
        for ent in self._format(predict).split(';'):
            # Some place names may contain commas, so don't check comma count here
            ent = self._format(ent)
            entity_pred.add(ent)
        return entity_truth, entity_pred

class EvaluatorRE(EvaluatorBase):
    def _init_metric(self):
        self.metric = MetricF1NA()  # Special handling for NA type relations

    def _init_audit(self):
        super()._init_audit()    
        self.audit += [
            AuditInvalid(),
            AuditFidelity(),
            AuditGoldenlabelFault(),
            AuditConfuseMatrix(),
            AuditNA()
        ]

    def _extract(self, json_data, predict):
        y_truth = set()
        # pattern = r'(?:head entity:)(.+?)(?:\s*,\s*)+(?:tail entity:)(.+?)(?:\s*,\s*)+(?:relation:)(.+?)(?:\s*,\s*)*$'
        for rel in self._format(json_data['Instance']['ground_truth']).split(';'):   # FIXME: field name may vary
            # Relations with type 'no_relation' or 'NA' are not ignored now, same below
            # elem = re.findall(pattern, rel)
            # if len(elem) == 0:
            #     continue
            # elem = ','.join(self._format(i) for i in elem[0])
            # elem = self._format(elem)
            elem = self._format(rel)
            if ':' not in elem:
                continue
            y_truth.add(elem)

        y_pred = set()
        # If model outputs 'no relation' or '[]', consider predicted relation set as empty, but no special handling needed here
        for rel in self._format(predict).split(';'):
            # Fields may contain commas themselves, so no count validation here
            # elem = re.findall(pattern, rel)
            # if len(elem) == 0:
            #     continue
            # elem = ','.join(self._format(i) for i in elem[0])
            # elem = self._format(elem)       # No problem that format can't solve, if there is, add more formats
            elem = self._format(rel)
            if ':' not in elem:
                continue
            y_pred.add(elem)
        return y_truth, y_pred

class EvaluatorMRC(EvaluatorBase):
    def _init_metric(self):
        self.metric = MetricF1()
    def _extract(self, json_data, predict):
        truth = self._remove_redundant_space(json_data['answer_text'])
        pred = self._remove_redundant_space(predict)
        return truth.lower(), pred.lower()


class EvaluatorSM(EvaluatorBase):
    def _init_metric(self):
        self.metric = MetricAcc()
    def _extract(self, json_data, predict):
        y_truth = self._remove_redundant_space(json_data['ground_truth'])
        y_pred = self._remove_redundant_space(predict)
        trans_dict = {
            'Yes': 'Yes',
            'No': 'No',
            'yes': 'Yes',
            'no': 'No'
        }
        if y_truth in trans_dict:
            y_truth = trans_dict[y_truth]
        if y_pred in trans_dict:
            y_pred = trans_dict[y_pred]
        return y_truth.lower(), y_pred.lower()

class EvaluatorEvent(EvaluatorBase):
    def _init_metric(self):
        self.metric = MetricAcc()
    def _extract(self, json_data, predict):
        y_truth = set()
        for event in self._resolve_brackets(json_data['Instance']['ground_truth']):   # FIXME: field name may vary
            event = self._format(event)
            event = event.replace('arguments:', '')
            event_elements = self._resolve_comma(event)  # Normalization of each pair needs to be done in advance because sorting follows
            
            event_string = ','.join(sorted(event_elements)) # 'a:b,c:d'
            y_truth.add(event_string)
        
        y_pred = set()
        for event in self._resolve_brackets(predict):
            event = self._format(event)
            event = event.replace('arguments:', '')
            event_elements = self._resolve_comma(event)  # Normalization of each pair needs to be done in advance because sorting follows
            
            event_string = ','.join(sorted(event_elements)) # 'a:b,c:d'
            y_pred.add(event_string)
        return y_truth, y_pred

class EvaluatorEET(EvaluatorBase):
    def _init_metric(self):
        self.metric = MetricAcc()
    def _extract(self, json_data, predict: str):
        y_truth = json_data['Instance']['ground_truth']
        y_truth = self._format(y_truth)

        y_pred = self._format(predict)
        return y_truth, y_pred

class EvaluatorEEA(EvaluatorBase):
    def _init_metric(self):
        self.metric = MetricF1()
    def _extract(self, json_data, predict: str):
        y_truth = set()
        for item in json_data['Instance']['ground_truth'].split(';'):
            if ':' not in item:
                continue
            y_truth.add(self._format(item))
        
        y_pred = set()
        for item in self._format(predict).split(';'):
            if ':' not in item:
                continue
            y_pred.add(self._format(item))
        
        return y_truth, y_pred

# The actual format later differs from the initial table, so the following tests may not pass, only as usage examples
if __name__ == '__main__':
    eval_ner = EvaluatorNER()
    eval_re = EvaluatorRE()
    eval_event = EvaluatorEvent()
    eval_mrc = EvaluatorMRC()
    eval_sm = EvaluatorSM()
    def test(evaluator:EvaluatorBase, json_str, predict):
        json_data = json.loads(json_str)
        evaluator.add(json_data, predict)
        print(evaluator.get_metric())
        print(evaluator.get_report())
    
    test(
        eval_ner,
        """
        [{
            "sentence": "I study at University A and want to go to City B for coffee.",
            "entities": [
                {
                    "name": "University A",
                    "type": "organization",
                    "pos": [
                        2,
                        6
                    ]
                },
                {
                    "name": "City B",
                    "type": "location",
                    "pos": [
                        11,
                        13
                    ]
                }
            ]
        }]
        """,
        ["organization: University A, location: City B"]
    )
    print('Expected Result: 1.0')

    test(
        eval_re,
        """
        [{
            "sentence": "I study at University A and want to go to City B for coffee.",
            "relations": [
                {
                    "head": 
                            {"name": "University A",
                                "type": "organization",
                                "pos": [
                                    2,
                                    6
                                ]
                            },
                    "type":
                            "no_relation",
                    "tail":
                            {"name": "City B",
                            "type": "location",
                            "pos": [
                                11,
                                13
                                ]
                            }
                }
            ]
        },
        {
            "sentence": "John studies at University A in City C.",
            "relations": [
                {
                    "head": 
                            {"name": "University A",
                                "type": "organization",
                                "pos": [
                                    6,
                                    10
                                ]
                            },
                    "type":
                            "locate in",
                    "tail":
                            {"name": "City C",
                            "type": "location",
                            "pos": [
                                3,
                                5
                                ]
                            }
                },
                {
                    "head": 
                            {"name": "John",
                                "type": "person",
                                "pos": [
                                    0,
                                    2
                                ]
                            },
                    "type":
                            "belong to",
                    "tail":
                            {"name": "University A",
                            "type": "organization",
                            "pos": [
                                6,
                                10
                                ]
                            }
                }
            ]
        }
        ]
        """,
        ['no relation', '(University A, locate  in, City C), (John,belong to, University A)']
    )
    print('Expected Result: 0.5')

    test(
        eval_event,
        """
        [{
            "sentence": "Company X laid off 4000 employees: when the times abandon you, they don't even say hello!",
            "events": [
                {
                    "trigger": "laid off",
                    "type": "organization-layoff", 
                    "pos":[
                        2,
                        3
                    ],
                    "arguments": [
                        {
                            "name": "Company X",
                            "role": "layoff_agent", 
                            "pos":[
                                0,
                                2
                            ]
                        }, 
                        {
                            "name": "4000 employees",
                            "role": "layoff_count", 
                            "pos":[
                                4,
                                9
                            ]
                        }
                    ]
                }
            ]
        }]
        """,
        ["(event type : organization-layoff, trigger: laid off, layoff_agent: Company X, layoff_count: 4000 employees), (event type : organization-layoff, trigger: laid off, layoff_agent: Company X, layoff_count: 3000 employees)"]
    )
    print('Expected Result: 2/3')
    test(
        eval_mrc,
        """
        [{
            "paragraph": "Another approach to brain function is to examine the consequences of damage to specific brain areas. Even though it is protected by the skull and meninges, surrounded by cerebrospinal fluid, and isolated from the bloodstream by the blood\u2013brain barrier, the delicate nature of the brain makes it vulnerable to numerous diseases and several types of damage. In humans, the effects of strokes and other types of brain damage have been a key source of information about brain function. Because there is no ability to experimentally control the nature of the damage, however, this information is often difficult to interpret. In animal studies, most commonly involving rats, it is possible to use electrodes or locally injected chemicals to produce precise patterns of damage and then examine the consequences for behavior.",
            "question": "What sare the benifts of the blood brain barrir?",
            "answer_start": 195,
            "answer_text": "isolated from the bloodstream"
        }]
        """,
        ["isolated from the bloodstream"]
    )
    print('Expected Result: 1.0')
    test(
        eval_sm,
        """
        [{
            "sen1": "A man with a hard hat is dancing.",
            "sen2": "A man wearing a hard hat is dancing.",
            "label": "Yes"
        },
        {
            "sen1": "Can installment payment be changed to interest-first payment?",
            "sen2": "Is there interest-first payment available?",
            "label": "No"
        }
        ]
        """,
        ["Yes", "no"]
    )
    print('Expected Result: 1.0')