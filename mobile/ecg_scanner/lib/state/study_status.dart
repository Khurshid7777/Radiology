// What the doctor sees for a study in the list (TZ addendum §18.8): one of a few plain states.
import '../upload/upload_queue.dart';

enum StudyStatus { draft, queued, uploading, processing, ready, retake, failed }

/// The study's state from the queue item and the last server answer (`scan`, may be null).
StudyStatus studyStatus(UploadItem item, Map<String, dynamic>? scan) {
  switch (item.state) {
    case UploadState.draft:
      return StudyStatus.draft;
    case UploadState.queued:
      return StudyStatus.queued;
    case UploadState.uploading:
      return StudyStatus.uploading;
    case UploadState.failed:
      return StudyStatus.failed;
    case UploadState.sent:
      final status = scan?['status'] as String?;
      if (status == null || status == 'queued' || status == 'processing') return StudyStatus.processing;
      if (status == 'rejected' || status == 'failed') return StudyStatus.retake;
      return StudyStatus.ready;
  }
}
