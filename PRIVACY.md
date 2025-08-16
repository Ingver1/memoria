# Privacy

The system exposes a `PRIVACY_MODE` environment variable to control how much
data is retained.

## Modes

- `strict` – stores only the minimum data required for operation. Features that
  could reveal identifiers, such as `/memory/mark_access`, are disabled.
- `standard` – enables additional features that may record limited extra data.

`PRIVACY_MODE` defaults to `strict` so the default configuration keeps the
minimal amount of data. Set `PRIVACY_MODE=standard` to opt into the extended
features.
