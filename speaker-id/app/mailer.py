# Copyright 2026 winoiknow (Eric Alborn, Anteon Group)
# Licensed under the Apache License, Version 2.0 (the "License").

"""Invitation email (Phase 5B) — on-site SMTP, with a console/log stub default.

The enrollment invite link is sent here. With no `SMTP_HOST` configured the email
is **logged** instead of sent (the stub) so the flow works end-to-end without a
relay; set `SMTP_*` to actually send. The body states the link's validity window
and gives recording guidance (quiet room, no other speakers, …) so the invitee
captures clean samples.
"""

from __future__ import annotations

import logging
import os
import smtplib
from email.message import EmailMessage

logger = logging.getLogger("speaker_id.mailer")

SMTP_HOST = os.environ.get("SMTP_HOST", "")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASS = os.environ.get("SMTP_PASS", "")
SMTP_FROM = os.environ.get("SMTP_FROM", "speaker-id@localhost")
SMTP_STARTTLS = os.environ.get("SMTP_STARTTLS", "1").strip().lower() in ("1", "true", "yes", "on")


def _body(speaker_name: str | None, link: str, ttl_hours: int) -> str:
    who = f" {speaker_name}" if speaker_name else ""
    return f"""Hello{who},

You've been invited to enroll your voice for speaker recognition.

Open this personal link and follow the steps (give consent, then record a few
short clips or upload them):

    {link}

This link is valid for {ttl_hours} hours. You can return to it as many times as
you like within that window to add more recordings.

For the best results, please:
  • Record in a quiet room — no background music, TV, or traffic noise.
  • Make sure no one else is talking; only your voice should be captured.
  • Speak naturally, as you normally would on a call.
  • Record about 5–10 seconds per clip, 3–5 clips total.
  • Use a decent microphone close to you (a headset is ideal).

If you didn't expect this invitation, you can ignore this email.

— speaker-id enrollment
"""


def send_invite_email(to_email: str | None, link: str, ttl_hours: int, speaker_name: str | None = None) -> bool:
    """Send (or, without SMTP configured, log) the invitation email.

    Returns True if handed to the relay (or logged in stub mode). Never raises —
    a mail failure must not break invite creation; the admin still gets the link
    back in the API response.
    """
    subject = "Voice enrollment invitation"
    body = _body(speaker_name, link, ttl_hours)

    if not SMTP_HOST:
        # Console/log stub — the link + body, so the flow is testable without SMTP.
        logger.warning("SMTP not configured — invite email NOT sent. Link for %s:\n%s",
                       to_email or "(no email)", link)
        logger.info("Invite email body (stub):\n%s", body)
        return True

    if not to_email:
        logger.warning("invite has no email address — cannot send")
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = SMTP_FROM
    msg["To"] = to_email
    msg.set_content(body)
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as s:
            if SMTP_STARTTLS:
                s.starttls()
            if SMTP_USER:
                s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
        logger.info("invite email sent to %s via %s:%d", to_email, SMTP_HOST, SMTP_PORT)
        return True
    except Exception as e:  # never break invite creation on a mail error
        logger.error("invite email to %s failed (%s) — admin can still share the link", to_email, e)
        return False
