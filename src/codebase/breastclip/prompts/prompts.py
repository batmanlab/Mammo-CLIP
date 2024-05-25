import random


def generate_report_from_labels(findings, prompt_json, deterministic=False):
    # Image and view column of vindr should be in order of ["CC", "MLO"]
    # CC_FINDING, MLO_FINDING are in order:
    # [[+ve right findings], [+ve left findings], [-ve right findings], [-ve left findings]]
    pos_right_findings, pos_left_findings, neg_right_findings, neg_left_findings = findings
    if "No Finding" in pos_right_findings or "No Finding" in pos_left_findings:
        neg_right_findings = []
        neg_left_findings = []

    report = []
    if len(pos_right_findings) > 0:
        for pos in pos_right_findings:
            cand = prompt_json[pos]["pos_right"]
            sentence = cand[0] if deterministic else random.choice(cand)
            if len(sentence) > 0:
                report.append(sentence)


    if len(pos_left_findings) > 0:
        for pos in pos_left_findings:
            cand = prompt_json[pos]["pos_left"]
            sentence = cand[0] if deterministic else random.choice(cand)
            if len(sentence) > 0:
                report.append(sentence)

    if len(neg_right_findings) > 0:
        for neg in neg_right_findings:
            cand = prompt_json[neg]["neg_right"]
            sentence = cand[0] if deterministic else random.choice(cand)
            if len(sentence) > 0:
                report.append(sentence)

    if len(neg_left_findings) > 0:
        for neg in neg_left_findings:
            cand = prompt_json[neg]["neg_left"]
            sentence = cand[0] if deterministic else random.choice(cand)
            if len(sentence) > 0:
                report.append(sentence)

    report = list(set(report))
    if not deterministic:
        random.shuffle(report)
    report = " ".join(report)
    return report
