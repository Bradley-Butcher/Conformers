from dataclasses import dataclass
from conformer.components.rejection import RejectionFunction
from conformer.components.admission import AdmissionFunction
from conformer.components.gcf import GroupConfidenceFunction
from conformer.components.fwer import FWERFunction

@dataclass
class Components:
    rejection: RejectionFunction = RejectionFunction()
    admission: AdmissionFunction = AdmissionFunction()
    group_confidence: GroupConfidenceFunction = GroupConfidenceFunction()
    FWER: FWERFunction = FWERFunction()