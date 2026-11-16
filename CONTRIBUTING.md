# Contributing

Thanks for helping with yue2-post.

By taking part you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Reporting a bug or asking for a feature

Open an [issue](https://github.com/Gitguy268/yue2-post/issues/new/choose) and use the template. Report security problems privately: see [SECURITY.md](SECURITY.md).

## Making a change

1. Fork the repository and create a branch from `main`.
2. Set it up:

   ```sh
   see the README: venv with the official YuE2 wheel, `hf download m-a-p/YuE2-3B --local-dir models/YuE2-3B`
   bash scripts/run_gen.sh
   ```

3. Make your change. Keep commits small, each with a clear message.
4. Run the checks below and update the README if behaviour changed.
5. Open a pull request and fill in the template.

## Checks

```sh
python3 -m compileall -q scripts
bash -n scripts/run_gen.sh
```
