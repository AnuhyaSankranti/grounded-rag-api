# Analytics Access Control

The application has four access groups. Admin users are developers and service owners
with access to all operational features. BraveHearts users are internal partner teams
who can preview features that are baking before broad launch. Unlimited users can use
all generally available features. Limited users can access only data fields approved
for their legal and business context.

The feature rollout order is Admin, BraveHearts, and then Unlimited. New functionality
is first validated by Admin users, then rolled out to BraveHearts for feedback, and
finally released to Unlimited users. Authorization is enforced by the backend for
every request; hiding a page in the frontend is not considered an access-control
boundary.
