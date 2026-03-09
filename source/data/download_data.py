import os.path
import sys

from pydicom.dataset import Dataset

from pynetdicom. ae import ApplicationEntity
from pynetdicom.sop_class import PatientRootQueryRetrieveInformationModelFind, PatientRootQueryRetrieveInformationModelGet, PatientRootQueryRetrieveInformationModelMove

from pynetdicom import AE, evt, build_role
from pynetdicom.sop_class import CTImageStorage, EnhancedCTImageStorage, XRay3DAngiographicImageStorage

import progressbar


def handle_store(event):
    """Handle a C-STORE request event."""
    ds = event.dataset
    ds.file_meta = event.file_meta

    # Save the dataset using the SOP Instance UID as the filename
    ds.save_as(ds.SOPInstanceUID, write_like_original=False)

    # Return a 'Success' status
    return 0x0000


def get_series(_assoc: ApplicationEntity, patient_id: str, study_instance_uid: str, series_instance_uid: str):
    ds = Dataset()
    ds.QueryRetrieveLevel = 'SERIES'
    ds.PatientID = patient_id
    ds.StudyInstanceUID = study_instance_uid
    ds.SeriesInstanceUID = series_instance_uid

    if _assoc.is_established:
        # Use the C-GET service to send the identifier
        # responses = _assoc.send_c_get(ds, PatientRootQueryRetrieveInformationModelGet)
        responses = _assoc.send_c_move(ds, "KLEBINGAT2", PatientRootQueryRetrieveInformationModelMove)
        for (status, identifier) in responses:
            if status:
                pass
                # print('C-GET query status: 0x{0:04x}'.format(status.Status))
            else:
                print('Connection timed out, was aborted or received invalid response')
                return get_series(get_assoc(), patient_id, study_instance_uid, series_instance_uid)
    else:
        return get_series(get_assoc(), patient_id, study_instance_uid, series_instance_uid)

    return _assoc


def find_dsa_series(_assoc: ApplicationEntity, series_description: str, study_instance_uid: str, series_number: str= '', study_date: str= ''):
    result = dict()

    ds = Dataset()
    ds.QueryRetrieveLevel = 'SERIES'
    ds.SeriesInstanceUID = ''
    ds.StudyInstanceUID = study_instance_uid
    ds.SeriesDescription = series_description
    ds.StudyDescription = ''
    ds.AccessionNumber = ''
    ds.SeriesNumber = series_number
    ds.StudyDate = study_date
    ds.PatientID = ''

    if _assoc.is_established:
        responses = _assoc.send_c_find(ds, PatientRootQueryRetrieveInformationModelFind)
        for (status, identifier) in responses:
            if status and status.Status == 0xFF00:
                result[identifier.SeriesInstanceUID] = {
                    "StudyInstanceUID": identifier.StudyInstanceUID,
                    "StudyDescription": identifier.StudyDescription,
                    "SeriesInstanceUID": identifier.SeriesInstanceUID,
                    "SeriesDescription": identifier.SeriesDescription,
                    "AccessionNumber": identifier.AccessionNumber,
                    "StudyDate": identifier.StudyDate,
                    "SeriesNumber": identifier.SeriesNumber,
                    "PatientID": identifier.PatientID,
                }
            elif not status:
                return find_dsa_series(get_assoc(), series_description, study_instance_uid, series_number, study_date)
    else:
        return find_dsa_series(get_assoc(), series_description, study_instance_uid, series_number, study_date)

    return result, _assoc


def get_ae():
    storage_classes = [CTImageStorage]

    _ae = AE()
    _ae.add_requested_context(PatientRootQueryRetrieveInformationModelFind)
    _ae.add_requested_context(PatientRootQueryRetrieveInformationModelGet)
    _ae.add_requested_context(PatientRootQueryRetrieveInformationModelMove)

    # ae.supported_contexts = AllStoragePresentationContexts
    for storage_class in storage_classes:
        _ae.add_requested_context(storage_class)

    return _ae


def get_assoc(_ae: AE = get_ae()):
    return _ae.associate("mp301idb1", 105, ae_title="NETGATE1_IDB1", ext_neg=[build_role(x, scp_role=True) for x in [CTImageStorage]], evt_handlers=[(evt.EVT_C_STORE, handle_store)])


if __name__ == "__main__":
    study_date = "20230101-"
    study_date = "20220101-20221231"
    # study_date = "20210101-20211231"
    # study_date = "20200101-20201231"
    # study_date = "20190101-20191231"
    # study_date = "20180101-20181231"
    # study_date = "20170101-20171231"
    # Use the C-GET service to send the identifier

    output_path = sys.argv[1] if len(sys.argv) > 1 else "/home/topf/dicom/ai-dsa/storescp"

    studies_sub, assoc = find_dsa_series(get_assoc(), series_description='*Sub Medium EE Auto*', study_instance_uid='', study_date=study_date)

    data = []

    print("Total: {}".format(len(studies_sub.values())))

    pb = progressbar.ProgressBar(max_value=len(studies_sub))

    for idx, series_sub in enumerate(studies_sub.values()):
        pb.update(idx+1)
        # print("{}/{}".format(idx+1, len(studies_sub.values())))
        patient_id = series_sub['PatientID']
        suid = series_sub['StudyInstanceUID']
        accNo = series_sub['AccessionNumber']
        if not accNo or len(accNo) == 0:
            accNo = suid
        series_number = series_sub['SeriesNumber']
        study_year = series_sub['StudyDate'][:4]

        if output_path:
            if os.path.exists(os.path.join(output_path, "{}/{}-{}".format(study_year, accNo, series_number))):
                # print("Skipping {}-{}: already existing".format(accNo, series_number))
                continue

        series_fill_dict, assoc = find_dsa_series(assoc, series_description='*Nat Fill Medium*', study_instance_uid=suid, series_number=series_number)

        series_fill_list = [series_fill for series_fill in series_fill_dict.values() if series_fill["SeriesNumber"] == series_number]

        if len(series_fill_list) == 1:
            series_fill = series_fill_list[0]
            series_instance_uid_sub = series_sub['SeriesInstanceUID']
            series_instance_uid_fill = series_fill['SeriesInstanceUID']
            data.append((series_sub, series_fill))

            assoc = get_series(assoc, patient_id, suid, series_instance_uid_sub)
            assoc = get_series(assoc, patient_id, suid, series_instance_uid_fill)
        else:
            print("error {}: {}".format(len(series_fill_dict), suid))
            pass

    assoc.release()
