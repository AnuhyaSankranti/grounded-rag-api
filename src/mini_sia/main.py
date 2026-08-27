import uvicorn

from mini_sia.api.app import create_app


app = create_app()


def run() -> None:
    uvicorn.run("mini_sia.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()

