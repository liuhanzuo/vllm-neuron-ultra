"""Keep the caller's serial ownership until physical drain is proven.

This module never closes or releases a lock, starts a service, or signals a PID.
The caller supplies ownership-checked cleanup and independent physical checks.
"""
def retain_until_drained(cleanup,verify,report_pending,await_retry):
    for callback in (cleanup,verify,report_pending,await_retry):
        if not callable(callback):raise TypeError('Explicit drain callbacks required')
    pending_reported=False
    while True:
        cleanup_error=None
        try:cleanup()
        except BaseException as error:cleanup_error=error
        try:
            proof=verify()
            if type(proof)is not dict or proof.get('physically_drained')is not True:
                raise RuntimeError('Independent actual physical drain proof required')
            return proof
        except BaseException as error:
            if not pending_reported:
                try:
                    report_pending(dict(cleanup_error=repr(cleanup_error),drain_error=repr(error),
                        serial_ownership_retained=True,automatic_successor_release=False))
                    pending_reported=True
                except BaseException:
                    # An unavailable output file is not permission to unlock.
                    pass
        # Even interruption of the wait must not unwind lock ownership.
        try:await_retry()
        except BaseException:pass
