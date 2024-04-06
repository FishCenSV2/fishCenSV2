#pragma once

#include <boost/asio.hpp>
#include <mutex>
#include <cstdint>
#include "server.hpp"

using boost::asio::ip::tcp;

class Session : public std::enable_shared_from_this<Session> {
public:
	std::vector<int> data_cpy;  ///< Copy of data to send to clients.
	unsigned request;					 ///< Amount of elements received by server.
	Server& server;						 ///< Reference to server object for copying image data.

	/** @brief Constructor for Session.
	*
	* @param socket A socket object (note object is moved).
	* @param server The server object creating the session.
	* @return None
	*/
	Session(tcp::socket socket, Server& server);

	/** @brief Starts the session by calling do_read()
	* 
	* @return None
	*/
	void start();

private:

	tcp::socket _socket; ///< Socket object

	/** @brief Reads data from the client
	*
	* @return None
	*/
	void do_read();

	/** @brief Writes image frame to client
	*
	* @return None
	*/
	void do_write();

};