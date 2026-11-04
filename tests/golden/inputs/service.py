"""Billing service."""

from dataclasses import dataclass


@dataclass
class Invoice:
    customer: str
    amount_cents: int

    def total_eur(self) -> float:
        """Convert cents to euros."""
        return self.amount_cents / 100


def make_invoice(customer: str, cents: int) -> Invoice:
    if cents < 0:
        raise ValueError("negative amount")
    return Invoice(customer, cents)
