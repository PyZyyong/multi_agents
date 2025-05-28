import os
import json
import httpx
import asyncio
from urllib.parse import urljoin

class QWeatherTool:
    """
    天气工具类, 使用和风天气API
    """
    def api_config(self):
        # https://dev.qweather.com/docs/api/warning/weather-warning/ 天气预警
        self.weather_warning_url = urljoin(self.base_url, "/v7/warning/now")
        # https://dev.qweather.com/docs/api/geoapi/city-lookup/ 城市查询
        self.get_location_url = urljoin(self.base_url, "/geo/v2/city/lookup")
        # https://dev.qweather.com/docs/api/weather/weather-daily-forecast/ {}天气预报
        self.get_daily_forecast_url = urljoin(self.base_url, "/v7/weather/{version}")

    def __init__(self):
        self.api_key = os.getenv("QWEATHER_API_KEY")
        self.base_url = os.getenv("QWEATHER_BASE_URL")
        if not self.api_key or not self.base_url:
            raise ValueError("天气API配置错误")
        self.api_config()

    async def get_location(self, city_name: str) -> str:
        """
        获取城市对应id
        """
        params = {
            "location": city_name,
            "key": self.api_key,
        }
        async with httpx.AsyncClient() as client:
            response = await client.get(
                self.get_location_url,
                params=params,
            )
            data = response.json()
            if data["code"] != "200":
                raise ValueError(f"获取城市id失败: {data}")
            return data["location"][0]["id"]

    async def get_weather_warning(self, city_name: str) -> str:
        """
        获取天气预警信息
        """
        location = await self.get_location(city_name)
        params = {
            "location": location,
            "key": self.api_key,
            "lang": "zh",
        }
        async with httpx.AsyncClient() as client:
            response = await client.get(
                self.weather_warning_url,
                params=params,
            )
            data = response.json()
            if data["code"] != "200":
                raise ValueError(f"获取天气预警失败: {data}")

            warnings = data.get("warning", [])
            if not warnings:
                return "当前没有天气预警信息"

            result = []
            for warning in warnings:
                result.append(
                    f"预警ID: {warning['id']}\n"
                    f"标题: {warning['title']}\n"
                    f"发布时间: {warning['pubTime']}\n"
                    f"开始时间: {warning['startTime']}\n"
                    f"结束时间: {warning['endTime']}\n"
                    f"预警类型: {warning['typeName']}\n"
                    f"预警等级: {warning['severity']} ({warning['severityColor']})\n"
                    f"发布单位: {warning['sender']}\n"
                    f"状态: {warning['status']}\n"
                    f"详细信息: {warning['text']}"
                )
            return"\n\n".join(result)

    async def get_daily_forecast(self, city_name: str, days: int = 3) -> str:
        """
        获取指定位置的天气预报
        参数:
            city_name: 城市名称
            days: 预报天数，可选值为 3、7、10、15、30，默认为 3
        返回:
            格式化的天气预报字符串
        """
        # 根据天数选择API版本
        version_map = {
            3: "3d", 7: "7d", 10: "10d", 15: "15d", 30: "30d",
        }
        version = version_map.get(days) or '3d'
        location = await self.get_location(city_name)
        params = {
            "location": location,
            "key": self.api_key,
            "lang": "zh"
        }
        async with httpx.AsyncClient() as client:
            response = await client.get(
                self.get_daily_forecast_url.format(version=version),
                params=params,
            )
            data = response.json()
            if data["code"] != "200":
                raise ValueError(f"获取天气预报失败: {data}")

            daily = data.get("daily", [])
            if not daily:
                return "无法获取天气预报信息"

            result = []
            for day in daily[:days]:  # 限制天数
                result.append(
                    f"日期: {day['fxDate']}\n"
                    f"日出: {day['sunrise']}  日落: {day['sunset']}\n"
                    f"最高温度: {day['tempMax']}°C  最低温度: {day['tempMin']}°C\n"
                    f"白天天气: {day['textDay']}  夜间天气: {day['textNight']}\n"
                    f"白天风向: {day['windDirDay']} {day['windScaleDay']}级 ({day['windSpeedDay']}km/h)\n"
                    f"夜间风向: {day['windDirNight']} {day['windScaleNight']}级 ({day['windSpeedNight']}km/h)\n"
                    f"相对湿度: {day['humidity']}%\n"
                    f"降水量: {day['precip']}mm\n"
                    f"紫外线指数: {day['uvIndex']}\n"
                    f"能见度: {day['vis']}km"
                )
            return"\n\n---\n\n".join(result)


qweather_tool = QWeatherTool()


if __name__ == "__main__":
    asyncio.run(qweather_tool.get_weather_warning("薛城区"))
    # print(asyncio.run(qweather_tool.get_daily_forecast("薛城区")))
