"""
Envelope dispatcher for parallel pipeline routing.

Routes EnvelopeV1 payloads to both the Action Item and Deal pipelines
concurrently, with isolated error handling per pipeline.
"""
