"""Outgoing email over SMTP (Resend in production; any SMTP provider works).

Two messages: the invite a person receives when the admin approves them, and the notice
the admins receive when someone requests access. Email is optional: with SMTP_HOST unset,
nothing is sent and the admin copies invite links by hand, as before.
"""

import html
import smtplib
import ssl
from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import formataddr

from invoice_agent.auth.store import INVITE_DAYS, User
from invoice_agent.config import Settings


class MailError(Exception):
    """An email could not be sent (bad settings, provider refused it, network down)."""


@dataclass(frozen=True)
class Email:
    """One message, with a plain-text and an HTML version."""

    to: str
    subject: str
    text: str
    html: str


class Mailer:
    """Send emails with the SMTP settings from the environment."""

    TIMEOUT_S = 20

    def __init__(self, settings: Settings) -> None:
        """Keep the settings.

        Args:
            settings: App settings; the smtp_* and mail_from* fields are used.
        """
        self.settings = settings

    @property
    def enabled(self) -> bool:
        """Whether email is configured.

        Returns:
            True if an SMTP host and a From address are set.
        """
        return bool(self.settings.smtp_host and self.settings.mail_from)

    def send(self, email: Email) -> None:
        """Send one email.

        Port 465 uses TLS from the start; any other port (normally 587) upgrades with
        STARTTLS. Certificates are always verified.

        Args:
            email: The message.

        Returns:
            None.

        Raises:
            MailError: If email isn't configured or sending fails.
        """
        s = self.settings
        if not self.enabled:
            raise MailError("Email is not configured")
        msg = EmailMessage()
        msg["From"] = formataddr((s.mail_from_name, s.mail_from))
        msg["To"] = email.to
        msg["Subject"] = email.subject
        msg.set_content(email.text)
        msg.add_alternative(email.html, subtype="html")
        context = ssl.create_default_context()
        try:
            if s.smtp_port == 465:
                with smtplib.SMTP_SSL(
                    s.smtp_host, s.smtp_port, timeout=self.TIMEOUT_S, context=context
                ) as smtp:
                    self._login_and_send(smtp, msg)
            else:
                with smtplib.SMTP(s.smtp_host, s.smtp_port, timeout=self.TIMEOUT_S) as smtp:
                    smtp.starttls(context=context)
                    self._login_and_send(smtp, msg)
        except (smtplib.SMTPException, OSError, ValueError) as e:
            raise MailError(f"{type(e).__name__}: {e}") from e

    def _login_and_send(self, smtp: smtplib.SMTP, msg: EmailMessage) -> None:
        """Log in (if a user is set) and hand the message to the server.

        Args:
            smtp: An open, encrypted SMTP connection.
            msg: The message.

        Returns:
            None.
        """
        if self.settings.smtp_user:
            smtp.login(self.settings.smtp_user, self.settings.smtp_password.get_secret_value())
        smtp.send_message(msg)


_FOOTER_TEXT = "Invoice Agent, a demo built by Santosh Ch."


def _wrap(body: str) -> str:
    """Put an HTML email body in a simple, email-client-safe layout.

    Args:
        body: Inner HTML (already escaped where it contains user input).

    Returns:
        A complete HTML document with inline styles.
    """
    return (
        '<!doctype html><html><body style="margin:0;background:#f5f7fb;'
        'font-family:Segoe UI,Helvetica,Arial,sans-serif;color:#101828">'
        '<div style="max-width:520px;margin:0 auto;padding:28px 16px">'
        '<div style="font-weight:700;font-size:17px;color:#4f46e5;margin-bottom:14px">'
        "Invoice Agent</div>"
        '<div style="background:#fff;border:1px solid #eaecf0;border-radius:14px;'
        'padding:24px;font-size:15px;line-height:1.55">'
        f"{body}</div>"
        f'<p style="color:#98a2b3;font-size:12px;text-align:center">{_FOOTER_TEXT}</p>'
        "</div></body></html>"
    )


def invite_email(user: User, invite_url: str) -> Email:
    """The email someone receives when their access request is approved.

    Args:
        user: The approved user.
        invite_url: Their one-time link to set a password.

    Returns:
        The Email.
    """
    name = user.name.split()[0] if user.name.strip() else "there"
    text = (
        f"Hi {name},\n\n"
        "Your request to try Invoice Agent has been approved.\n\n"
        f"Set your password here (the link works once and expires in {INVITE_DAYS} days):\n"
        f"{invite_url}\n\n"
        "Then upload a sample invoice and see the extracted, verified data. Please use "
        "sample or synthetic invoices only: the demo runs on free AI models whose "
        "providers may log what they receive.\n\n"
        f"Santosh Ch\n\n{_FOOTER_TEXT}\n"
    )
    url = html.escape(invite_url, quote=True)
    body = (
        f"<p>Hi {html.escape(name)},</p>"
        "<p>Your request to try <b>Invoice Agent</b> has been approved.</p>"
        f'<p style="margin:22px 0"><a href="{url}" style="background:#4f46e5;color:#fff;'
        "text-decoration:none;font-weight:600;padding:11px 20px;border-radius:10px;"
        'display:inline-block">Set your password</a></p>'
        f'<p style="color:#667085;font-size:13px">The link works once and expires in '
        f"{INVITE_DAYS} days. If the button doesn't work, copy this address:<br>"
        f'<span style="word-break:break-all">{url}</span></p>'
        '<p style="color:#667085;font-size:13px">Please use sample or synthetic invoices only: '
        "the demo runs on free AI models whose providers may log what they receive.</p>"
        "<p>Santosh Ch</p>"
    )
    return Email(user.email, "Your access to Invoice Agent", text, _wrap(body))


def request_notice(
    to: str, name: str, email: str, company: str, reason: str, admin_url: str
) -> Email:
    """The email an admin receives when someone asks for access.

    Args:
        to: The admin's address.
        name: The requester's name.
        email: The requester's email.
        company: Their company, or ''.
        reason: What they want to try, or ''.
        admin_url: Link to the admin page.

    Returns:
        The Email.
    """
    lines = [f"Name: {name}", f"Email: {email}"]
    if company:
        lines.append(f"Company: {company}")
    if reason:
        lines.append(f"Wants to: {reason}")
    text = (
        "Someone requested access to Invoice Agent.\n\n"
        + "\n".join(lines)
        + f"\n\nApprove or reject: {admin_url}\n"
    )
    rows = "".join(
        f'<tr><td style="color:#667085;padding:4px 12px 4px 0;vertical-align:top">'
        f"{html.escape(k)}</td><td>{html.escape(v)}</td></tr>"
        for k, v in (
            ("Name", name),
            ("Email", email),
            ("Company", company),
            ("Wants to", reason),
        )
        if v
    )
    body = (
        "<p>Someone requested access to <b>Invoice Agent</b>.</p>"
        f'<table style="font-size:14px;border-collapse:collapse">{rows}</table>'
        f'<p style="margin:22px 0 0"><a href="{html.escape(admin_url, quote=True)}" '
        'style="background:#4f46e5;color:#fff;text-decoration:none;font-weight:600;'
        'padding:10px 18px;border-radius:10px;display:inline-block">Open the admin page</a></p>'
    )
    one_line = " ".join(name.split())[:60]  # form input: no line breaks in a header
    return Email(to, f"Access request: {one_line}", text, _wrap(body))
